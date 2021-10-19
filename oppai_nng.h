#pragma once

// Requires c++20 and AVX2
// Clang seems to generate better AVX over MSVC at a glance in the compiler explorer. Might want to check that out, -std=c++20 -mavx2 -mfma -O2

// Not particularly focused on stable parity. Could change in the future.

#include <fstream>
#include <string_view>
#include <array>
#include <tuple>
#include <type_traits>
#include <vector>
#include <thread>
#include <charconv>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MM_ALIGN __declspec(align(32))

#include "AVX.h"

namespace ocpp {

	constexpr bool low_memory_usage{ 1 }; // Having this true sped up within temptation by 2-3x on my machine

	constexpr bool AVX_angle_calc{ 1 }; // Huge performance win
	constexpr bool AVX_strain_sum{ 1 }; // Slightly changes accuracy, but takes it from 2*0.23ms -> 2*0.012ms on my machine

	// After 829 iterations weight gets denormalized, which in most cases is treated as a zero
	// 832 just allows it to neatly fit in the avx batch
	constexpr size_t strain_weight_max_interation{ AVX_strain_sum ? 832 : 829 };

	constexpr bool skip_strain_length_total{ 1 }; // Affects an un-used param?

	//Occlude low strain values, inaccuracy liberties lead to the strain data to be quite noisy so a 0 compare is mute.
	constexpr float strain_cut_off{0.5e-3f};

	constexpr size_t TIMING_CHUNKS{ 64 }, NOTE_CHUNKS{ 512 };

	constexpr size_t DIFF_SPEED{ 0 }, DIFF_AIM{ 1 };

	constexpr float SINGLE_SPACING{ 125.0f },
					STAR_SCALING_FACTOR{ 0.0675f }, /* star rating multiplier */
					EXTREME_SCALING_FACTOR{ 0.5f }, /* used to mix aim/speed stars */
					STRAIN_STEP{ 400.0f }, /* diffcalc uses peak strains of 400ms chunks */
					DECAY_WEIGHT{ 0.9f }, /* peak strains are added in a weighed sum */
					MAX_SPEED_BONUS{ 45.0f }, /* ~330BPM 1/4 streams */
					MIN_SPEED_BONUS{ 75.0f }, /* ~200BPM 1/4 streams */
					ANGLE_BONUS_SCALE{ 90.f },
					AIM_TIMING_THRESHOLD{ 107.f },
					SPEED_ANGLE_BONUS_BEGIN{ 5 * M_PI / 6.f },
					AIM_ANGLE_BONUS_BEGIN{ M_PI / 3.f };

	constexpr float decay_base[] = { 0.3f, 0.15f }; /* strains decay per interval */
	constexpr float weight_scaling[] = { 1400.0f, 26.25f }; /* balances aim/speed */

	using namespace std::literals;	

	template <typename F> struct on_scope_exit {
		private: F func;
		public:
			on_scope_exit(const on_scope_exit&) = delete;
			on_scope_exit& operator=(const on_scope_exit&) = delete;
			on_scope_exit(F&& f) : func(std::forward<F>(f)) {}
			~on_scope_exit() { func(); }
	};

	#define PCAT0(x, y) PCAT1(x, y)
	#define PCAT1(x, y) PCAT2(!, x ## y)
	#define PCAT2(x, r) r
	#define ON_SCOPE_EXIT(...) on_scope_exit PCAT0(__scope, __LINE__) {[&]{__VA_ARGS__}}

	#define ERR(s, ...) printf(s, __VA_ARGS__)

	#define push_vector_chunk(vector, chunk_size, value){\
		const auto s{ vector.size() };\
		if(vector.capacity() == s)\
			vector.reserve(s + chunk_size);\
		vector.push_back(value);\
	}

	template<typename T> T pow2(const T& v) { return v * v; }

	typedef unsigned char u8;
	typedef unsigned short u16;
	typedef unsigned int u32;
	typedef unsigned long long u64;

	enum hitobject_type {
		OBJ_CIRCLE = 1 << 0, 
		OBJ_SLIDER = 1 << 1,
		OBJ_SPINNER1 = 1 << 3
	};

	struct vec2{

		float x, y;

		float length() const{
			const auto p2{ x * x + y * y };
			return p2 > 0.f ? sqrtf(p2) : 0.f;

		}

		float length2() const {
			return x * x + y * y;
		}

		vec2 operator*(const float f) const{
			return vec2{ x * f, y * f };
		}

		vec2 operator-(const vec2&__restrict B) __restrict const noexcept{
			return vec2{ x - B.x, y - B.y };
		}

		float operator[](const size_t i) const { return i ? y : x;}

		bool is_zero() const { return x == 0.f && y == 0.f; }

	};
	
	float dot(const vec2& A, const vec2& B){
		return A.x * B.x + A.y * B.y;
	}

	struct _timing {

		u32 change;        /* if 0, ms_per_beat is -100.0f * sv_multiplier */

		float time;        /* milliseconds */
		float ms_per_beat;
		float px_per_beat;

		/* taiko stuff */
		float beat_len;
		float velocity;

	};

	struct _hit_object { // Pretty memory chunky

		// These are floats to ease implicit casting for floating arithmetic, at the cost of some memory

		u32 type : 4, repeats : 16 /*repeats might be 15, check it later*/, timing_index : 12;
		float time;
		float duration;
		vec2 pos;
		vec2 norm_pos;
		float angle;
		
		float distance;

		int get_combo_count(const float px_per_beat, const float tick_rate)__restrict const noexcept{

			const float repetitions{ float(repeats) };

			const float num_beats{ repetitions * distance * px_per_beat };

			int ticks = (int)ceil(
				(num_beats - 0.1f/*I feel this is way too much*/) / repetitions * tick_rate
			);

			--ticks;

			ticks *= repeats;     /* account for repetitions */
			ticks += repeats + 1; /* add heads and tails */

			/*
			 * actually doesn't include first head because we already
			 * added it by setting res = nobjects
			 */

			return std::max(0, ticks - 1);

		}


	};

	#define MODS_NOMOD 0
	#define MODS_NF (1<<0)
	#define MODS_EZ (1<<1)
	#define MODS_TD (1<<2)
	#define MODS_HD (1<<3)
	#define MODS_HR (1<<4)
	#define MODS_SD (1<<5)
	#define MODS_DT (1<<6)
	#define MODS_RX (1<<7)
	#define MODS_HT (1<<8)
	#define MODS_NC (1<<9)
	#define MODS_FL (1<<10)
	#define MODS_AT (1<<11)
	#define MODS_SO (1<<12)
	#define MODS_AP (1<<13)
	#define MODS_PF (1<<14)
	#define MODS_KEY4 (1<<15) /* TODO: what are these abbreviated to? */
	#define MODS_KEY5 (1<<16)
	#define MODS_KEY6 (1<<17)
	#define MODS_KEY7 (1<<18)
	#define MODS_KEY8 (1<<19)
	#define MODS_FADEIN (1<<20)
	#define MODS_RANDOM (1<<21)
	#define MODS_CINEMA (1<<22)
	#define MODS_TARGET (1<<23)
	#define MODS_KEY9 (1<<24)
	#define MODS_KEYCOOP (1<<25)
	#define MODS_KEY1 (1<<26)
	#define MODS_KEY3 (1<<27)
	#define MODS_KEY2 (1<<28)
	#define MODS_SCOREV2 (1<<29)
	#define MODS_TOUCH_DEVICE MODS_TD
	#define MODS_NOVIDEO MODS_TD /* never forget */
	#define MODS_SPEED_CHANGING (MODS_DT | MODS_HT | MODS_NC)
	#define MODS_MAP_CHANGING (MODS_HR | MODS_EZ | MODS_SPEED_CHANGING)

	struct _score_stats {

		float acc_percent;
		u32 mods;
		u16 c_300, c_100, c_50, c_miss;
		u16 combo;

		float aim_pp, speed_pp, acc_pp, pp;

	};

	_inline float get_acc_percent(u32 c300, u32 c100, u32 c50, u32 cMiss) {

		const auto sum{ c300 + c100 + c50 + cMiss };

		return sum > 0 ? float(c50 * 50 + c100 * 100 + c300 * 300) / float(sum * 300) : 0.f;

	}

	float acc_calc(int n300, int n100, int n50, int misses) {
		int total_hits = n300 + n100 + n50 + misses;
		float acc = 0;
		if (total_hits > 0) {
			acc = ((n50 * 50.0f) + (n100 * 100.0f) + (n300 * 300.0f))
				/ (total_hits * 300.0f);
		}
		return acc;
	}

	_inline float get_acc_percent(const _score_stats& sc) {
		
		const auto sum{ (int)sc.c_300 + (int)sc.c_100 + (int)sc.c_50 + (int)sc.c_miss };

		return sum > 0 ? float(50 * sc.c_50 + 100 * sc.c_100 + 300 * sc.c_300) / float(sum * 300) : 0.f;

	}

	struct _pp_meta {

		enum : u8 {
			parse_general = 0,
			parse_difficulty,
			parse_timingpoints,
			parse_hitobjects
		};

		u32 parse_mode : 2, version:5, mode:2, strain_spawn_thread:1;

		struct {
			float HP, CS, OD, AR, slider_multiplier, slider_tickrate;
		} diff;

		u32 mods;
		u16 max_combo, c_note, c_slider, c_spinner;
		float total_stars;

		float speed_stars, speed_diff;
		float aim_stars, aim_diff;

		std::vector<_hit_object> objects;
		std::vector<_timing> timing;

		std::vector<MM_ALIGN float> speed_highest_strains;
		std::vector<MM_ALIGN float> aim_highest_strains;

		float interval_end;
		float max_strain[2];
		float speed_mul;

		char buff[0xFFFF];

		void acc_round_normal(float acc_percent, int misses, _score_stats& ss) {

			const int nobjects{ int(objects.size()) };

			const int max300 = nobjects - misses;

			misses = std::min(nobjects, misses);

			acc_percent = std::clamp(acc_percent, 0.17f,
				std::max(0.17f,get_acc_percent(max300, 0, 0, misses))
			);

			float n100 = 1.5f * ((1.f - acc_percent) * (float)nobjects + (float)misses);
			
			const float over_100 = n100 / (float)(nobjects);
			
			float n50 = 4.f * std::max(over_100 - 1.f, 0.f) * nobjects;
			
			n100 = n50 > 0.5f ? n100 / over_100 - n50 : n100;
			
			n100 = round(n100);
			n50 = round(n50);
			
			int n300 = nobjects - (int)n100 - (int)n50 - misses;
			
			ss.c_300 = std::max(n300, 0);
			ss.c_100 = std::max((int)n100, 0);
			ss.c_50 = std::max((int)n50, 0);
			ss.c_miss = misses;

		}

		void acc_round(float acc_percent, int misses, _score_stats& ss) { 
			acc_round_normal(acc_percent * 0.01f,misses, ss);
		}

	};


	struct _batch_strain {

		MM_ALIGN __m256 angle;
		MM_ALIGN __m256 distance;
		MM_ALIGN __m256 delta;

		// Gross :(
		MM_ALIGN __m256 p_distance;
		MM_ALIGN __m256 p_delta;

		std::array<u16, 8> note_id{};

	};

	struct _batch_return {

		MM_ALIGN __m256 decay_speed;
		MM_ALIGN __m256 decay_aim;
		MM_ALIGN __m256 strain_speed;
		MM_ALIGN __m256 strain_aim;

	};

	_batch_return calc_strain_batch(_batch_strain&__restrict bs) {
		
		_batch_return ret;

		// The play speed multiplier is pre applied to bs.delta.

		{

			const auto decay_delta = _mm256_mul_ps(_mm256_set1_ps(0.001f), bs.delta);

			//ret.decay_speed = pow_8(_mm256_set1_ps(decay_base[DIFF_SPEED]), decay_delta);
			//ret.decay_aim = pow_8(_mm256_set1_ps(decay_base[DIFF_AIM]), decay_delta);

			ret.decay_speed = intel_exp_8(_mm256_mul_ps(_mm256_set1_ps(-1.20397282f), decay_delta));
			ret.decay_aim = intel_exp_8(_mm256_mul_ps(_mm256_set1_ps(-1.89712000f), decay_delta));

		}

		const __m256 strain_time = _mm256_max_ps(bs.delta, _mm256_set1_ps(50.f));

		// Speed
		{

			const auto speed_distance = _mm256_min_ps(bs.distance, _mm256_set1_ps(SINGLE_SPACING));
			const auto speed_delta = _mm256_max_ps(bs.delta, _mm256_set1_ps(MAX_SPEED_BONUS));

			__m256 speed_bonus = _mm256_set1_ps(0.f);

			{
				const auto min_speed_bonus = _mm256_set1_ps(MIN_SPEED_BONUS);

				const auto mask = _mm256_cmp_ps(speed_delta, min_speed_bonus, _CMP_LT_OQ);

				if (_mm256_movemask_ps(mask)) {// Perf. win on slower map

					constexpr auto exp{ 1.f / 40.f };

					const auto base{ _mm256_sub_ps(min_speed_bonus, speed_delta) };

					const auto add = pow2_8(_mm256_mul_ps(base, _mm256_set1_ps(exp)));

					speed_bonus = _mm256_blendv_ps(_mm256_set1_ps(0.f), add, mask);

				}

			}

			__m256 angle_bonus = _mm256_set1_ps(0.f);

			#define AND(x,y) _mm256_and_ps((x),(y))

			{
				const auto min_speed_angle_bonus{ _mm256_set1_ps(SPEED_ANGLE_BONUS_BEGIN) };	

				const auto mask0 = _mm256_cmp_ps(bs.angle, min_speed_angle_bonus, _CMP_LT_OQ);// Could add the original nan check here
				auto mask1 = _mm256_cmp_ps(bs.angle, _mm256_set1_ps((float)M_PI * 0.5f), _CMP_LT_OQ);
				auto mask2 = AND(
						_mm256_cmp_ps(speed_distance, _mm256_set1_ps(ANGLE_BONUS_SCALE), _CMP_LT_OQ),
						_mm256_cmp_ps(bs.angle, _mm256_set1_ps((float)M_PI * 0.25f), _CMP_LT_OQ)
					);
				auto mask3 = _mm256_cmp_ps(speed_distance, _mm256_set1_ps(ANGLE_BONUS_SCALE), _CMP_LT_OQ);

				mask1 = AND(mask1, mask0);

				mask2 = AND(mask2, mask1);
				mask3 = _mm256_andnot_ps(mask1, mask3);

				const auto s = nv_sin_8(_mm256_mul_ps(_mm256_set1_ps(1.5), _mm256_sub_ps(min_speed_angle_bonus, bs.angle)));

				// Believe in the heart of the compiler.

				//TODO: benchmark branching
				const auto bonus3/*mul with bonus2*/ =					
					_mm256_movemask_ps(mask3) == 0 ? _mm256_set1_ps(0.f) :
						nv_sin_8(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(M_PI * 0.5f), bs.angle), _mm256_set1_ps(1.27323954472)));

				const auto bonus2 =
					_mm256_movemask_ps(mask2) == 0 ? _mm256_set1_ps(0.f) :
						_mm256_mul_ps(
							_mm256_set1_ps(0.28f * 0.1f),
							_mm256_min_ps(_mm256_sub_ps(_mm256_set1_ps(ANGLE_BONUS_SCALE), speed_distance), _mm256_set1_ps(1.f))
						);

				const auto bonus0 = _mm256_mul_ps(pow2_8(s), _mm256_set1_ps(1.f / 3.57f));

				const auto bonus1 = _mm256_set1_ps(0.28f);

				angle_bonus = _mm256_blendv_ps(angle_bonus, bonus0, mask0);
				angle_bonus = _mm256_blendv_ps(angle_bonus, bonus1, mask1);
				angle_bonus = _mm256_blendv_ps(angle_bonus, bonus2, mask2);
				angle_bonus = _mm256_blendv_ps(angle_bonus, _mm256_mul_ps(bonus3, bonus2), mask3);
				angle_bonus = _mm256_add_ps(angle_bonus, _mm256_set1_ps(1.f));
			}

			#undef AND

			const auto p0 = _mm256_mul_ps(angle_bonus, _mm256_add_ps(_mm256_set1_ps(1.f), _mm256_mul_ps(_mm256_set1_ps(0.75f), speed_bonus)));
			const auto p1 = _mm256_add_ps( _mm256_set1_ps(0.95f),
				_mm256_mul_ps(
					_mm256_add_ps(speed_bonus, _mm256_set1_ps(1.f)),
					pow_3p5_8(_mm256_mul_ps(speed_distance, _mm256_set1_ps(1.f / SINGLE_SPACING)))
				)
			);

			ret.strain_speed = _mm256_div_ps(_mm256_mul_ps(p0, p1), strain_time);

		}

		// Aim
		{

			const auto prev_strain_delta = _mm256_max_ps(_mm256_set1_ps(50.f), bs.p_delta);

			auto weighted_distance = pow_8(bs.distance, _mm256_set1_ps(0.99f));

			{
				const auto invalid_mask = _mm256_cmp_ps(bs.distance, _mm256_set1_ps(0.f), _CMP_LE_OQ);
				weighted_distance = _mm256_blendv_ps(weighted_distance, _mm256_set1_ps(0.f), invalid_mask);
			}

			__m256 result;

			{

				__m256 angle_bonus{ _mm256_set1_ps(0.f) };

				{
					const auto p0 = _mm256_max_ps(_mm256_sub_ps(bs.p_distance, _mm256_set1_ps(ANGLE_BONUS_SCALE)), _mm256_set1_ps(0.f));
					const auto p1 = nv_sin_8(_mm256_sub_ps(bs.angle, _mm256_set1_ps(AIM_ANGLE_BONUS_BEGIN)));
					const auto p2 = _mm256_max_ps(_mm256_sub_ps(bs.distance, _mm256_set1_ps(ANGLE_BONUS_SCALE)), _mm256_set1_ps(0.f));
					
					angle_bonus = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_mul_ps(p0, p2), pow2_8(p1)));
				}


				auto p1 = pow_8(_mm256_max_ps(angle_bonus, _mm256_set1_ps(0.f)), _mm256_set1_ps(0.99f));
				{
					const auto invalid_mask = _mm256_cmp_ps(angle_bonus, _mm256_set1_ps(0.f), _CMP_LE_OQ);
					p1 = _mm256_blendv_ps(p1, _mm256_set1_ps(0.f), invalid_mask);
				}

				const auto p2 = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_max_ps(bs.p_delta, _mm256_set1_ps(AIM_TIMING_THRESHOLD)));

				const auto p3 = _mm256_mul_ps(_mm256_mul_ps(p1, _mm256_set1_ps(1.5f)), p2);

				result = 
					_mm256_blendv_ps(_mm256_set1_ps(0.f), p3, _mm256_cmp_ps(bs.angle, _mm256_set1_ps(AIM_ANGLE_BONUS_BEGIN), _CMP_GT_OQ));

			}

			const __m256 vel = _mm256_div_ps(weighted_distance, _mm256_max_ps(_mm256_set1_ps(AIM_TIMING_THRESHOLD), strain_time));

			ret.strain_aim = _mm256_max_ps(_mm256_add_ps(result, vel), _mm256_div_ps(weighted_distance, strain_time));

		}	

		ret.strain_speed = _mm256_mul_ps(ret.strain_speed, _mm256_set1_ps(weight_scaling[DIFF_SPEED]));
		ret.strain_aim = _mm256_mul_ps(ret.strain_aim, _mm256_set1_ps(weight_scaling[DIFF_AIM]));

		return ret;
	}

	_inline void d_update_max_strains(_pp_meta& pp, float cur_time, float prev_time, const vec2& speed_strain, const vec2& aim_strain){


		/* make previous peak strain decay until the current obj */
		while (cur_time > pp.interval_end) {

			if(pp.max_strain[0] > strain_cut_off)
				pp.speed_highest_strains.push_back(pp.max_strain[0]);
			if (pp.max_strain[1] > strain_cut_off)
				pp.aim_highest_strains.push_back(pp.max_strain[1]);

			{
				
				constexpr float base[2]{-1.20397282f/*log(0.3f)*/, -1.89712000f/*log(0.15f)*/ };

				const auto d = (pp.interval_end - prev_time) * 0.001f;

				pp.max_strain[0] = speed_strain[1] * exp(d * base[0]);
				pp.max_strain[1] = aim_strain[1] * exp(d * base[1]);

			}

			pp.interval_end += STRAIN_STEP * pp.speed_mul;

		}
	
		pp.max_strain[0] = std::max(pp.max_strain[0], speed_strain[0]);
		pp.max_strain[1] = std::max(pp.max_strain[1], aim_strain[0]);

	}
	
	void d_weigh_individual(std::vector<float>& __restrict vec, float& star, float& diff) {

		std::sort(vec.begin(), vec.end(), std::greater<float>());// This is very slow

		float total{};
		float difficulty{};
		float weight{ 1.f };

		const size_t size{ std::min(vec.size(), strain_weight_max_interation)};
		const size_t real_size{ vec.size() };

		size_t i{};

		if constexpr (AVX_strain_sum) {

			__m256 sum{}, sum2{};
			
			MM_ALIGN std::array<float, 8> _weight;

			for (; i + 8 <= size ; i += 8) {
			
				const auto strains = _mm256_load_ps(&vec[i]);

				#define DO(x) { _weight[x] = weight; weight *= DECAY_WEIGHT; }

				DO(0); DO(1); DO(2); DO(3); DO(4); DO(5); DO(6); DO(7);

				const auto _diff = _mm256_mul_ps(strains, _mm256_load_ps(_weight.data()));

				sum2 = _mm256_add_ps(sum2, _diff);

				if constexpr (skip_strain_length_total == 0) {
					const auto _star = pow_8(strains, _mm256_set1_ps(1.2f));
					sum = _mm256_add_ps(sum, _star);
				}

				#undef DO

			}

			difficulty = _256_HADD(sum2);

			if constexpr (skip_strain_length_total == 0){

				for (; i + 8 <= real_size; i += 8) {
					const auto _star = pow_8(_mm256_load_ps(&vec[i]), _mm256_set1_ps(1.2f));
					sum = _mm256_add_ps(sum, _star);
				}
			}

			total = _256_HADD(sum);

		}

		{

			for ( ; i < size; ++i) {

				const auto& strain = vec[i];

				if constexpr (skip_strain_length_total == 0)
					total += (float)pow(strain, 1.2);

				difficulty += strain * weight;
				weight *= DECAY_WEIGHT;

			}

			if constexpr (skip_strain_length_total == 0) {

				for (; i < real_size; ++i)
					total += (float)pow(vec[i], 1.2);

			}

		}

		star = difficulty;
		diff = total;

	}

	void d_weigh_strains(_pp_meta& pp) {

		/* sort strains from highest to lowest */

		if (pp.strain_spawn_thread && 0) [[unlikely]] {

			// Runs substantialy slower on my machine, I don't want there to be a co thread always running so at least on my machine the latency hit is too hard
			// Even if the thread was spawned at the start of load_map execution it still is much slower!

			std::thread t{ d_weigh_individual, std::ref(pp.aim_highest_strains), std::ref(pp.aim_stars), std::ref(pp.aim_diff)};

			d_weigh_individual(pp.speed_highest_strains, pp.speed_stars, pp.speed_diff);

			t.join();

		}else {
			
			d_weigh_individual(pp.speed_highest_strains, pp.speed_stars, pp.speed_diff);
			d_weigh_individual(pp.aim_highest_strains, pp.aim_stars, pp.aim_diff);

		}
	}

	void calc_strain(_pp_meta& pp){
		
		{
			const auto res{ 1 + (int)ceil((pp.objects.back().time - pp.objects[0].time) / (STRAIN_STEP * pp.speed_mul)) };

			pp.speed_highest_strains.reserve(res);
			pp.aim_highest_strains.reserve(res);

		}

		pp.interval_end = (
			(float)ceil(pp.objects[0].time / (STRAIN_STEP * pp.speed_mul))
			* STRAIN_STEP * pp.speed_mul
			);

		/* this implementation doesn't account for sliders*/

		_batch_strain batch;// 192 bytes

		u8 c{};

		const float rspeed = 1.f / pp.speed_mul;

		const _hit_object* last{ &pp.objects[0] };

		float previous_distance{}, previous_delta{};

		float p_speed_strain{}, p_aim_strain{};

		float last_time{ last->time }; float last_pdistance{};

		//MM_ALIGN __m256 max_strain[2]{}; was going to use to help with sorting

		for (size_t i{1}, size{ pp.objects.size() }; i < size; ++i) {

			const auto& n = pp.objects[i];

			//if ((n.type & (OBJ_CIRCLE | OBJ_SLIDER)) == 0) continue;		

			batch.note_id[c] = i;
			batch.angle.m256_f32[c] = n.angle;// Might want to blit this properly.

			batch.p_delta.m256_f32[c] = previous_delta;

			batch.delta.m256_f32[c] = (previous_delta = ((n.time - last->time) * rspeed));
			batch.distance.m256_f32[c] = (previous_distance = (n.norm_pos - last->norm_pos).length2());

			last = &n;

			if (++c == 8 || i + 1 == size) {

				batch.distance = _mm256_sqrt_ps(batch.distance);

				batch.p_distance.m256_f32[0] = last_pdistance;

				memcpy(&batch.p_distance.m256_f32[1], &batch.distance.m256_f32[0], sizeof(batch.distance) - sizeof(float));

				last_pdistance = batch.distance.m256_f32[7];

				const auto&__restrict ret = calc_strain_batch(batch);

				for (size_t z{}; z < c; ++z) {

					const float speed_strain = p_speed_strain * ret.decay_speed.m256_f32[z] + ret.strain_speed.m256_f32[z];
					const float aim_strain = p_aim_strain * ret.decay_aim.m256_f32[z] + ret.strain_aim.m256_f32[z];

					const auto c_time = pp.objects[batch.note_id[z]].time;

					d_update_max_strains(pp, c_time, last_time, { speed_strain, p_speed_strain }, { aim_strain, p_aim_strain });

					last_time = c_time;
					p_speed_strain = speed_strain;
					p_aim_strain = aim_strain;
				}

				c = 0;

			}

		}

		pp.speed_highest_strains.push_back(pp.max_strain[0]);
		pp.aim_highest_strains.push_back(pp.max_strain[1]);

		d_weigh_strains(pp);

		pp.speed_stars = sqrt(pp.speed_stars) * STAR_SCALING_FACTOR;
		pp.aim_stars = sqrt(pp.aim_stars) * STAR_SCALING_FACTOR;

		pp.total_stars = pp.aim_stars + pp.speed_stars +
			(float)fabs(pp.speed_stars - pp.aim_stars) * EXTREME_SCALING_FACTOR;

	}



	_inline bool get_line(std::string_view& __restrict range, std::string_view& __restrict out) {

		for (size_t i{}, size{ range.size() }; i < size; ++i) {

			if (range[i] == '\n') [[unlikely]] {

				out = std::string_view(range.data(), i);
				range = std::string_view(range.data() + i + 1, range.size() - (i + 1));

				return 1;
			}
		}

		return 0;
	}

	template<char delim, typename ...T>
	 const std::tuple<T...>& split_line(std::string_view line, std::tuple<T...>& v) {

		size_t i{}, size{ line.size() }, start{};

		std::apply(
			[&](auto&... value_pack){
				(([&](auto& value){

					using _t = std::remove_cvref<decltype(value)>::type;

					if (i >= size)
						return;

					while (i < size && line[i] != delim) {
						++i;
						continue;
					}

					if constexpr (std::is_same<_t, std::string_view>::value) {
						value = std::string_view(line.data() + start, line.data() + (i >= size ? size : i));
					} else std::from_chars(line.data() + start, line.data() + (i >= size ? size : i), value);

					start = std::min(++i, size - 1);
		
				}(value_pack)), ...);

			}, v
		);

		return v;
	}

	 #define PARSE(str, ret)\
		if(line.starts_with(str##sv)){\
			decltype(ret) t;/*extra copy for bitwise members*/\
			std::from_chars(line.data() + sizeof(str) - 1, line.data() + line.size(), t);\
			ret = t;\
			return;\
		}

	void parse_general(_pp_meta& pp, std::string_view line) {

		PARSE("osu file format v", pp.version);
		PARSE("Mode: ", pp.mode);

	}

	void parse_difficulty(_pp_meta& pp, std::string_view line) {

		PARSE("HPDrainRate:", pp.diff.HP);
		PARSE("CircleSize:", pp.diff.CS);
		PARSE("OverallDifficulty:", pp.diff.OD);
		PARSE("ApproachRate:", pp.diff.AR);
		PARSE("SliderMultiplier:", pp.diff.slider_multiplier);
		PARSE("SliderTickRate:", pp.diff.slider_tickrate);

	}

	void parse_timing(_pp_meta& pp, std::string_view line) {

		using timing_point = std::tuple<int, float, int, u8, u8, u8, u8, u8>;

		timing_point temp;

		std::get<6>(temp) = (u8)1;// Set the timing point to be an uninherited node by default

		const auto& [time, beat_ms, meter, sampleSet, sampleIndex, volume, uninherited, effect]{ split_line<','>(line, temp) };

		_timing t{};

		t.time = (float)time;
		t.ms_per_beat = beat_ms;
		t.change = uninherited;

		push_vector_chunk(pp.timing, TIMING_CHUNKS, t);

	}

	void parse_note(_pp_meta& pp, std::string_view line) {

		using hit_object = std::tuple<int, int, int, u8, u8, std::string_view, int, float, std::string_view>;

		hit_object temp;

		const auto& [x, y, time, type, hitSound, slider_param,repeats, distance, edgesounds] { split_line<','>(line, temp) };

		_hit_object ho{};

		ho.time = (float)time;
		
		ho.type = type;
		ho.pos = vec2((float)x, (float)y);
		
		if (type & OBJ_SPINNER1) {
			++pp.c_spinner;// Would prefer a stack increment
			ho.pos = { 256.f, 192.f };

		}else if (type & OBJ_SLIDER) {

			++pp.c_slider;
			ho.repeats = repeats;
			ho.distance = distance;

		}else ++pp.c_note;

		push_vector_chunk(pp.objects, NOTE_CHUNKS, ho);

	}


	#undef PARSE

	void parse_line(_pp_meta& pp, std::string_view line) {	

		size_t l_size{ line.size() };

		if (l_size && line[l_size - 1] == '\r')// Clip windows return carriage
			line = std::string_view(line.data(), l_size -= 1);

		if (l_size == 0)
			return;

		if (line[0] == '[') [[unlikely]] {
			
			if (line == "[Difficulty]"sv)
				pp.parse_mode = pp.parse_difficulty;

			if (line == "[TimingPoints]"sv)
				pp.parse_mode = pp.parse_timingpoints;

			if (line == "[HitObjects]"sv)
				pp.parse_mode = pp.parse_hitobjects;

			if (line == "[Colours]"sv)
				pp.parse_mode = 0;

			return;
		}

		if(pp.parse_mode == pp.parse_hitobjects)
			parse_note(pp, line);
		else if (pp.parse_mode == pp.parse_timingpoints)
			parse_timing(pp, line);
		else if (pp.parse_mode == pp.parse_difficulty)
			parse_difficulty(pp, line);
		else
			parse_general(pp, line);

	}

	u8 load_map(_pp_meta& pp, const std::string_view map_path/*must be null terminated!*/) {

		pp.objects.reserve(NOTE_CHUNKS);
		pp.timing.reserve(TIMING_CHUNKS);

		pp.diff.AR = -1.f;
		pp.parse_mode = pp.parse_general;
		
		{
			std::ifstream file(map_path.data(), std::ios::binary | std::ios::ate | std::ios::in);

			if (file.is_open() == 0) [[unlikely]]
				return 1;

			ON_SCOPE_EXIT( file.close(); );

			const auto file_size{ (size_t)file.tellg() };

			file.seekg(0, std::ios::beg);

			if constexpr (low_memory_usage) {

				for (size_t c{}; c < file_size; ) {

					const auto read_size{ std::min(file_size - c, sizeof pp.buff) };

					file.read(pp.buff, read_size);

					std::string_view _memory(pp.buff, read_size);

					for (std::string_view line; get_line(_memory, line);)
						parse_line(pp, line);

					if (_memory.data() == pp.buff) {
						// Single line was larger than the pp.buff length
						// Might want to handle this better in the future
						ERR("pp.buff size is too small for a line!\n");
						return 1;
					}

					// If just read to the end of the file we can break out
					if (c + read_size == file_size) {
						parse_line(pp, _memory);
						break;
					}

					// Otherwise read from start of the last new line
					file.seekg((c += size_t(_memory.data() - pp.buff)), std::ios::beg);
				}

			} else {

				std::vector<char> buff;
				std::string_view _memory;

				if (sizeof(pp.buff) >= file_size)
					(_memory = std::string_view(pp.buff, file_size), file.read(pp.buff, file_size));
				else {

					buff.resize(file_size);
					file.read(buff.data(), file_size);

					_memory = std::string_view(buff.data(), file_size);
				}

				for (std::string_view line; get_line(_memory, line);)
					parse_line(pp, line);

				parse_line(pp, _memory);

			}
		}

		pp.diff.AR = pp.diff.AR == -1.f ? pp.diff.OD : pp.diff.AR;

		{ // Initialize timing

			float last_ms{};

			for (size_t i{}, size{ pp.timing.size() }; i < size; ++i) {

				auto& __restrict /*non conforming*/ t = pp.timing[i];

				last_ms = t.change ? t.ms_per_beat : last_ms;

				float sv_multiplier{
					(!t.change && t.ms_per_beat < 0) ? -100.0f / t.ms_per_beat : 1.0f
				};

				t.beat_len = std::clamp(last_ms / sv_multiplier, 10.f, 1000.f);
				t.px_per_beat = pp.diff.slider_multiplier * 100.0f;

				t.velocity = t.px_per_beat / t.beat_len;

				sv_multiplier = pp.version >= 8 ? sv_multiplier : 1.f;

				t.beat_len *= sv_multiplier;
				t.px_per_beat *= sv_multiplier;

				t.px_per_beat = 1.f / t.px_per_beat;
				t.velocity = 1.f / t.velocity;

			}

		}

		if (pp.objects.size() == 0 || pp.timing.size() == 0)
			return 0;

		pp.speed_mul = 1.f;

		if(pp.mods & MODS_MAP_CHANGING){

			if (pp.mods & (MODS_DT | MODS_NC))
				pp.speed_mul = 1.5f;
			if (pp.mods & MODS_HT)
				pp.speed_mul = 0.75f;

			float cs_mul{ 1.f };
			float od_mul{ 1.f };

			if (pp.mods & MODS_HR) {
				od_mul = 1.4f;
				cs_mul = 1.3f;
			}
			if (pp.mods & MODS_EZ) {
				od_mul = 0.5f;
				cs_mul = 0.5f;
			}

			pp.diff.CS = std::clamp(pp.diff.CS * cs_mul, 0.f,10.f);
			pp.diff.OD *= od_mul;

			constexpr float od10_ms[] = { 20, 20 }; /* std, taiko */
			constexpr float od0_ms[] = { 80, 50 };
			constexpr float od_ms_step[] = { 6.0f, 3.0f };

			// This is changing the OD even at 1x play speed, intentional?
			float odms = od0_ms[0] - (float)ceil(od_ms_step[0] * pp.diff.OD);
			odms = std::clamp(odms, od10_ms[0], od0_ms[0]);
			odms /= pp.speed_mul;
			pp.diff.OD = (od0_ms[0] - odms) / od_ms_step[0];


		}

		{
			pp.max_combo = pp.objects.size();

			const float radius{
				(512.f / 16.0f) * (1.0f - 0.7f * (pp.diff.CS - 5.0f) / 5.0f)
			};

			const float scaling_factor = 52.0f / radius;
			
			const auto timing_count{ pp.timing.size() };

			u32 timing_index{};

			float next_timing_node{ timing_count > 1 ? pp.timing[1].time : FLT_MAX };

			const auto& _time_ref = pp.timing;
			const auto adv_timing = [timing_count, &_time_ref](const float c_time, float&__restrict next, u32&__restrict c_index) {

				for (; c_time >= next ;) {
					c_index = std::min<u32>(c_index + 1, timing_count - 1);
					next = (c_index == timing_count - 1 ? FLT_MAX : _time_ref[c_index + 1].time);
				}

				return c_index;
			};


			if constexpr(AVX_angle_calc == 0){

				for (size_t i{}, size{ pp.objects.size() }; i < size; ++i) {

					auto& __restrict/*non conforming*/ cur{ pp.objects[i] };

					cur.norm_pos = cur.pos * scaling_factor;

					if (i >= 2) {

						{
							auto& prev1 = pp.objects[i - 1];
							auto& prev2 = pp.objects[i - 2];

							vec2 v1 = prev2.norm_pos - prev1.norm_pos;
							vec2 v2 = cur.norm_pos - prev1.norm_pos;

							const float _dot = dot(v1, v2);
							const float _det = v1.x * v2.y - v1.y * v2.x;

							cur.angle = (float)fabs(atan2(_det, _dot));

							if (cur.angle > M_PI)
								cur.angle -= M_PI;

							cur.timing_index = adv_timing(cur.time, next_timing_node, timing_index);;

						}

					} else {
						cur.angle = 0.f; //get_nan();
					}

				}

			} else {

				_hit_object* const start_ptr = &pp.objects[0];
				_hit_object* const end_ptr = &pp.objects.back();

				for (size_t i{}, size{ pp.objects.size() }; i < size; i += 8) {

					_hit_object* start{ start_ptr + i };

					MM_ALIGN std::array<std::array<float, 8>, 4> uv;

					for (size_t z{}; z < 8; ++z) {// TODO: see if its worth vecorizing this

						_hit_object* cur = std::min((start + z), end_ptr);

						cur->norm_pos = cur->pos * scaling_factor;

						const _hit_object* prev1 = std::max(cur - 1, start_ptr);
						const _hit_object* prev2 = std::max(cur - 2, start_ptr);

						vec2 v1 = prev2->norm_pos - prev1->norm_pos;
						vec2 v2 = cur->norm_pos - prev1->norm_pos;

						vec2 v3 = vec2(1.f,1.f) - vec2(0.5f,0.5f);

						const u32 is_inf{ u32(v1.length2() == 0.f) + (u32(v2.length2() == 0.f) << 1) };

						if (is_inf) [[unlikely]]{
							if (is_inf == 3)
								v2 = (v1 = vec2{ 1.f,0.f });
							else if (is_inf == 2)
								v2 = v1;
							else
								v1 = v2;
						}

						uv[0][z] = v1.x;
						uv[1][z] = v1.y;
						uv[2][z] = v2.x;
						uv[3][z] = v2.y;

					}

					MM_ALIGN /*redundant*/ std::array<float, 8> res;

					_mm256_store_ps(res.data(), nv_acos_8(dot_4_8(uv)));// There is probably a more efficent optimization where the angle is actually used

					for (size_t z{}; z < 8; ++z) {

						auto& c = pp.objects[std::min(i + z, size - 1)];

						c.angle = res[z];												

						c.timing_index = adv_timing(c.time, next_timing_node, timing_index);

						if ((c.type & OBJ_SLIDER) == 0)
							continue;

						const auto& t{ pp.timing[timing_index] };

						c.duration = c.distance * float(c.repeats) * t.velocity;

						pp.max_combo += c.get_combo_count(t.px_per_beat, pp.diff.slider_tickrate);

					}

				}

			}

		}

		calc_strain(pp);

	}

	float base_pp(float stars) {
		return (float)pow(5.0f * std::max(1.0f, stars / 0.0675f) - 4.0f, 3.0f)
			/ 100000.0f;
	}


	float calc_pp_8(_pp_meta& pp, const std::array<_score_stats*, 8> scores) {



		return 0;
	}

	float calc_pp_single(_pp_meta& pp, _score_stats& score) {

		const int ncircles{ pp.c_note };
		const int nobjects{ (int)pp.objects.size() };

		float nobjects_over_2k = float(nobjects) / 2000.0f;

		float length_bonus = (
			0.95f +
			0.4f * std::min(1.0f, nobjects_over_2k) +
			(nobjects > 2000 ? (float)log10(nobjects_over_2k) * 0.5f : 0.0f)
			);

		float miss_penality_aim = 0.97 * pow(1 - pow((double)score.c_miss / (double)nobjects, 0.775), score.c_miss);
		float miss_penality_speed = 0.97 * pow(1 - pow((double)score.c_miss / (double)nobjects, 0.775f), pow(score.c_miss, 0.875f));

		float combo_break = ( (float)pow(score.combo, 0.8f) / (float)pow(pp.max_combo, 0.8f));

		float accuracy = get_acc_percent(score.c_300, score.c_100, score.c_50, score.c_miss);
		float real_acc = get_acc_percent(std::max(0, int(score.c_300) - int(pp.c_slider) - int(pp.c_spinner)), score.c_100, score.c_50, score.c_miss);

		/* ar bonus -------------------------------------------------------- */
		float ar_bonus = 0.0f;

		if (pp.diff.AR > 10.33f) {/* high ar bonus */
			ar_bonus += 0.4f * (pp.diff.AR - 10.33f);
		}else if (pp.diff.AR < 8.0f) { /* low ar bonus */
			ar_bonus += 0.01f * (8.0f - pp.diff.AR);
		}

		score.aim_pp = base_pp(pp.aim_stars);
		score.aim_pp *= length_bonus;
		if (score.c_miss > 0) {
			score.aim_pp *= miss_penality_aim;
		}
		score.aim_pp *= combo_break;
		score.aim_pp *= 1.0f + (float)std::min(ar_bonus, ar_bonus * (nobjects / 1000.0f));

		/* hidden */
		float hd_bonus = 1.0f;
		if (score.mods & MODS_HD) {
			hd_bonus += 0.04f * (12.0f - pp.diff.AR);
		}

		score.aim_pp *= hd_bonus;

		/* flashlight */
		if (score.mods & MODS_FL) {
			float fl_bonus = 1.0f + 0.35f * std::min(1.0f, nobjects / 200.0f);
			if (nobjects > 200) {
				fl_bonus += 0.3f * std::min(1.f, (nobjects - 200) / 300.0f);
			}
			if (nobjects > 500) {
				fl_bonus += (nobjects - 500) / 1200.0f;
			}
			score.aim_pp *= fl_bonus;
		}

		/* acc bonus (bad aim can lead to bad acc) */
		float acc_bonus = 0.5f + accuracy / 2.0f;

		/* od bonus (high od requires better aim timing to acc) */
		float od_squared = (float)pow(pp.diff.OD, 2);
		float od_bonus = 0.98f + od_squared / 2500.0f;

		score.aim_pp *= acc_bonus;
		score.aim_pp *= od_bonus;

		/* speed pp -------------------------------------------------------- */
		score.speed_pp = base_pp(pp.speed_stars);
		score.speed_pp *= length_bonus;
		if (score.c_miss > 0) {
			score.speed_pp *= miss_penality_speed;
		}
		score.speed_pp *= combo_break;
		if (pp.diff.AR > 10.33f) {
			score.speed_pp *= 1.0f + (float)std::min(ar_bonus, ar_bonus * (nobjects / 1000.0f));;
		}

		score.speed_pp *= hd_bonus;

		/* scale the speed value with accuracy slightly */
		score.speed_pp *= (0.95f + od_squared / 750) * (float)pow(accuracy, (14.5 - std::max(pp.diff.OD, 8.f)) / 2);

		/* it's important to also consider accuracy difficulty when doing that */
		score.speed_pp *= (float)pow(0.98f, score.c_50 < nobjects / 500.0f ? 0.00 : score.c_50 - nobjects / 500.0f);

		/* acc pp ---------------------------------------------------------- */
		/* arbitrary values tom crafted out of trial and error */
		score.acc_pp = (float)pow(1.52163f, pp.diff.OD) *
			(float)pow(real_acc, 24.0f) * 2.83f;

		/* length bonus (not the same as speed/aim length bonus) */
		score.acc_pp *= std::min(1.15f, (float)pow(ncircles / 1000.0f, 0.3f));

		if (score.mods & MODS_HD) score.acc_pp *= 1.08f;
		if (score.mods & MODS_FL) score.acc_pp *= 1.02f;

		/* total pp -------------------------------------------------------- */
		float final_multiplier = 1.12f;

		if (score.mods & MODS_NF) final_multiplier *= (float)std::max(0.9f, 1.0f - 0.2f * score.c_miss);
		if (score.mods & MODS_SO) final_multiplier *= 1.0 - pow((double)pp.c_spinner / nobjects, 0.85);

		score.pp = (float)(
			pow(
				pow(score.aim_pp, 1.1f) +
				pow(score.speed_pp, 1.1f) +
				pow(score.acc_pp, 1.1f),
				1.0f / 1.1f
			) * final_multiplier
			);

		return score.pp;
	}

	#undef push_vector_chunk
	#undef PCAT0
	#undef PCAT1
	#undef PCAT2
	#undef ON_SCOPE_EXIT
}
