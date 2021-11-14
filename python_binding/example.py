import nng_oppai

mods = 0
map_path = './path_to_map.osu'

pp_ptr = nng_oppai.load_map(f'{mods}|{map_path}')

acc = 100
miss_count = 0
max_combo = -1

print('pp: ', nng_oppai.calc_pp_single(f'{pp_ptr}|{mods}|{acc}|{miss_count}|{max_combo}'))

print('map stats', nng_oppai.get_beatmap_stats(pp_ptr))

nng_oppai.free_map(pp_ptr) # Free the pp_ptr once you are done