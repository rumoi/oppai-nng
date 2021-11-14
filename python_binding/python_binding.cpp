#include <Python.h> //Linked to python3.9
#include "oppai_ngg.h"

#define from_sv(sv, out) (std::from_chars(sv.data(), sv.data() + sv.size(), out))

struct _pyref {
    PyObject* obj; PyObject* operator()() { return obj; }
    ~_pyref() { Py_XDECREF(obj); }
};

void write_buffer(char*& dst, std::string_view src, char* end){
    const auto size{ std::min(src.size(), size_t(end - dst)) };
    memcpy(dst, src.data(), size);
    dst += size;
}

PyObject* py_get_beatmap_stats(PyObject* _module, PyObject* arg) {

    ocpp::_pp_meta* pp{ (ocpp::_pp_meta*)PyLong_AsSize_t(PyNumber_Long(arg)) };

    char buff[512];
    char* c{buff + 1}, *end{ buff + sizeof(buff) - 1 };

    buff[0] = '{';

    #define WRITE(name, value) write_buffer(c, "\n\t\""#name"\": ", end); c = std::to_chars(c, end, value).ptr;

    WRITE(accuracy, pp->diff.OD);
    WRITE(ar, pp->diff.AR);
    WRITE(cs, pp->diff.CS);
    WRITE(drain, pp->diff.HP);

    WRITE(total_stars, pp->total_stars);
    WRITE(speed_stars, pp->speed_stars);
    WRITE(aim_stars, pp->aim_stars);

    WRITE(beatmapset_id, pp->set_id);
    WRITE(beatmap_id, pp->beatmap_id);

    #undef WRITE

    write_buffer(c, "\n}", end);
    *c = 0;

    return PyUnicode_FromString(buff);
}

PyObject* py_load_map(PyObject* _module, PyObject* _map_path) {

    _pyref map_str{
        PyUnicode_AsEncodedString(_pyref{ PyObject_Repr(_map_path) }(), "ascii", 0)
    };

    ocpp::_pp_meta* pp{ new ocpp::_pp_meta{} };

    auto raw{ std::string_view(PyBytes_AS_STRING(map_str())) };
    raw = raw.substr(1, raw.size() - 2);

    const auto mid{ raw.find_first_of('|') };

    {
        const auto mods = raw.substr(0, mid);
        from_sv(mods, pp->mods);
    }

    char map_path[260]{};
    {
        const auto raw_map_path = raw.substr(mid + 1);
        memcpy(map_path, raw_map_path.data(), std::min(raw_map_path.size(), (size_t)sizeof(map_path) - 1));
    }
    
    ocpp::load_map(*pp, map_path);

    return PyLong_FromSize_t((size_t)pp);
}

PyObject* py_free_map(PyObject* _module, PyObject* arg) {

    _pyref conv{ PyNumber_Long(arg) };

    auto* pp{ (ocpp::_pp_meta*)PyLong_AsSize_t(conv()) };

    if(pp) delete pp;

    return PyBool_FromLong(pp != 0);
}


PyObject* py_calc_pp_single(PyObject* _module, PyObject* _map_path) {

    _pyref repr{ PyObject_Repr(_map_path) },
        str{ PyUnicode_AsEncodedString(repr(), "ascii", 0) };

    auto raw = std::string_view(PyBytes_AS_STRING(str()));
    raw = raw.substr(1, raw.size() - 2);

    using _stats = std::tuple<size_t, unsigned int, float, int, int>;

    _stats temp{};
    const auto [pp_ptr, mods, acc, miss_count, max_combo] = ocpp::split_line_AVX<'|'>(raw, temp);

    ocpp::_pp_meta* pp = (ocpp::_pp_meta*)pp_ptr;

    ocpp::_score_stats stats{};

    pp->acc_round(acc, miss_count, stats);
    //stats.acc_percent is not currently used
    stats.mods = mods;
    stats.combo = max_combo == -1 ? pp->max_combo : max_combo;

    return PyFloat_FromDouble(ocpp::calc_pp_single(*pp, stats));

}

static PyMethodDef method_table[] {

    { "load_map", py_load_map, METH_O, "Allocates a new map object and loads the map data" },
    { "calc_pp_single", py_calc_pp_single, METH_O, 0 },
    { "free_map", py_free_map, METH_O, "Deallocates a map object"},
    { "get_beatmap_stats", py_get_beatmap_stats, METH_O, 0},

    {}
};

static PyModuleDef nng_oppai_module = {
    PyModuleDef_HEAD_INIT,
    "nng_oppai",// Module name
    "osu! pp calculator",// Module desc
    0,//Module size
    method_table
};

PyMODINIT_FUNC PyInit_nng_oppai() {
    return PyModule_Create(&nng_oppai_module);
}