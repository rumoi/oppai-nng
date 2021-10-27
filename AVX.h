#pragma once

enum _compiler_version{ MSVC = 0, CLANG, GCC };

#if defined(__clang__)
	constexpr auto compiler_version = CLANG;
#elif defined(__GNUC__) || defined(__GNUG__)
	constexpr auto compiler_version = GCC;
#elif defined(_MSC_VER)
	constexpr auto compiler_version = MSVC;
#endif


#include <bit>
#include <immintrin.h>
#include <array>

//#define _256_HADD(x)(\
//	x.m256_f32[0] + x.m256_f32[1] + x.m256_f32[2] + x.m256_f32[3] +\
//	x.m256_f32[4] + x.m256_f32[5] + x.m256_f32[6] + x.m256_f32[7])

#define _256_HADD(x) [x]{ MM_ALIGN float _v[8]; _mm256_store_ps(_v, x); return _v[0] + _v[1] + _v[2] + _v[3] + _v[4] + _v[5] + _v[6] + _v[7]; }();

#define MM256_SET1(x)_mm256_set1_ps(x)

#ifdef _MSC_VER 

__forceinline __m256 operator+(const __m256& x, const __m256& y) { return _mm256_add_ps(x, y); }
__forceinline __m256 operator-(const __m256& x, const __m256& y) { return _mm256_sub_ps(x, y); }
__forceinline __m256 operator*(const __m256& x, const __m256& y) { return _mm256_mul_ps(x, y); }
__forceinline __m256 operator/(const __m256& x, const __m256& y) { return _mm256_div_ps(x, y); }

__forceinline __m256 operator+(const float x, const __m256& y) { return _mm256_add_ps(MM256_SET1(x), y); }
__forceinline __m256 operator-(const float x, const __m256& y) { return _mm256_sub_ps(MM256_SET1(x), y); }
__forceinline __m256 operator-(const __m256& y, const float x) { return _mm256_sub_ps(y, MM256_SET1(x)); }
__forceinline __m256 operator*(const float x, const __m256& y) { return _mm256_mul_ps(MM256_SET1(x), y); }
__forceinline __m256 operator/(const float x, const __m256& y) { return _mm256_div_ps(MM256_SET1(x), y); }

#endif

_inline __m256 pow2_8(__m256 x) { return x * x; }
_inline __m256 abs_8(__m256 x) { return _mm256_and_ps(x, MM256_SET1(std::bit_cast<float>(0x7FFFFFFF))); }
_inline __m256 rsign_8(__m256 x) { return _mm256_xor_ps(x, MM256_SET1(std::bit_cast<float>(0x80000000))); }

__m256 nv_sin_8(__m256 x) {

	//https://developer.download.nvidia.com/cg/sin.html

	struct float4 { float x, y, z, w; };

	constexpr float4 c0 = float4{ 0.f, 0.5f, 1.f, 0.f };
	constexpr float4 c1 = float4{ 0.25f, -9.0f, 0.75f, 0.159154943091f };
	constexpr float4 c2 = float4{ 24.9808039603f, -24.9808039603f, -60.1458091736f, 60.1458091736f };
	constexpr float4 c3 = float4{ 85.4537887573f, -85.4537887573f, -64.9393539429f, 64.9393539429f };
	constexpr float4 c4 = float4{ 19.7392082214f, -19.7392082214f, -1.f, 1.f };

	__m256 r0_x, r0_y, r0_z;
	__m256 r1_x, r1_y, r1_z;
	__m256 r2_x, r2_y, r2_z;

	r1_x = _mm256_fmadd_ps(x, MM256_SET1(c1.w), MM256_SET1(-c1.x));// only difference from cos!

	r1_y = r1_x - _mm256_floor_ps(r1_x);// and extract fraction

	r2_x = _mm256_and_ps(MM256_SET1(1.f), _mm256_cmp_ps(r1_y, MM256_SET1(c1.x), _CMP_LT_OQ));// range check: 0.0 to 0.25
	r2_y = _mm256_and_ps(MM256_SET1(1.f), _mm256_cmp_ps(r1_y, MM256_SET1(c1.y), _CMP_GT_OQ));
	r2_z = _mm256_and_ps(MM256_SET1(1.f), _mm256_cmp_ps(r1_y, MM256_SET1(c1.z), _CMP_GT_OQ));// range check: 0.75 to 1.0

	{
		const auto inv = MM256_SET1(std::bit_cast<float>(0x80000000));

		r2_y = _mm256_xor_ps(_mm256_fmadd_ps(MM256_SET1(c4.z), r2_z, c4.w * r2_y) + c4.z * r2_x, inv);// range check: 0.25 to 0.75
		r2_x = _mm256_xor_ps(r2_x, inv);
		r2_z = _mm256_xor_ps(r2_z, inv);

	}

	r0_x = pow2_8(c0.x - r1_y);
	r0_y = pow2_8(c0.y - r1_y);
	r0_z = pow2_8(c0.z - r1_y);// range centering

	{// start power series
		r1_x = _mm256_fmadd_ps(r0_x, MM256_SET1(c2.x), MM256_SET1(c2.z));
		r1_y = _mm256_fmadd_ps(r0_y, MM256_SET1(c2.y), MM256_SET1(c2.w));
		r1_z = _mm256_fmadd_ps(r0_z, MM256_SET1(c2.x), MM256_SET1(c2.z));
	}

	#define DO(_x,_y,_z)\
		r1_x = _mm256_fmadd_ps(r1_x, r0_x, MM256_SET1(_x));\
		r1_y = _mm256_fmadd_ps(r1_y, r0_y, MM256_SET1(_y));\
		r1_z = _mm256_fmadd_ps(r1_z, r0_z, MM256_SET1(_z));

	DO(c3.x, c3.y, c3.x);
	DO(c3.z, c3.w, c3.z);
	DO(c4.x, c4.y, c4.x);
	DO(c4.z, c4.w, c4.z);

	#undef DO
	
	//return _mm256_fmadd_ps(r1_z, r2_z, (r1_y * r2_y) + (r1_x * r2_x));
	return (r1_z * r2_z) + (r1_y * r2_y) + (r1_x * r2_x);// range extract
}

__m256 dot_4_8(const std::array<std::array<float, 8>, 4>& __restrict/*must be 32 aligned*/ x) {
	
	auto X1 = _mm256_load_ps(&x[0][0]);
	auto Y1 = _mm256_load_ps(&x[1][0]);
	auto X2 = _mm256_load_ps(&x[2][0]);
	auto Y2 = _mm256_load_ps(&x[3][0]);

	{// Normalize input vectors, approximated

		const auto D1 = _mm256_rsqrt_ps(_mm256_fmadd_ps(Y1, Y1, X1 * X1));
		const auto D2 = _mm256_rsqrt_ps(_mm256_fmadd_ps(Y2, Y2, X2 * X2));

		X1 = X1 * D1;
		Y1 = Y1 * D1;

		X2 = X2 * D2;
		Y2 = Y2 * D2;

	}

	auto DOT = _mm256_fmadd_ps(X1, X2, Y1 * Y2);
	
	// Clamp between -1 and 1
	DOT = _mm256_min_ps(DOT, MM256_SET1(1.f));

	return _mm256_max_ps(DOT, MM256_SET1(-1.f));
}

__m256 nv_acos_8(__m256 x) {

	// https://developer.download.nvidia.com/cg/acos.html

	const auto negate = _mm256_and_ps(MM256_SET1(1.f), _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ));

	x = _mm256_and_ps(x, MM256_SET1(std::bit_cast<float>(0x7FFFFFFF)));

	auto y = _mm256_fmadd_ps(x, MM256_SET1(-0.0187293f), MM256_SET1(0.0742610f));

	y = _mm256_fmadd_ps(x, y, MM256_SET1(-0.2121144f));
	y = _mm256_fmadd_ps(x, y, MM256_SET1(1.5707288f)) * _mm256_sqrt_ps(1.f - x);

	y = _mm256_fmadd_ps(y, -2.f * negate, y);

	return _mm256_fmadd_ps(negate, MM256_SET1((float)M_PI), y);
}

__m256 nv_atan2_8(__m256 y, __m256 x){

	//https://developer.download.nvidia.com/cg/atan2.html

	const auto x_abs = _mm256_and_ps(x, MM256_SET1(std::bit_cast<float>(0x7FFFFFFF)));;
	const auto y_abs = _mm256_and_ps(y, MM256_SET1(std::bit_cast<float>(0x7FFFFFFF)));;
	
	const auto invalid = _mm256_and_ps(
		_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_EQ_OQ),
		_mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_EQ_OQ)
	);

	const auto r = _mm256_min_ps(x_abs, y_abs) / _mm256_max_ps(x_abs, y_abs);//dbz

	const auto sqr = r * r;

	#define DO(x) t0 = _mm256_fmadd_ps(t0, sqr, MM256_SET1(x))

	auto t0 = _mm256_fmadd_ps(MM256_SET1(-0.013480470f), sqr, MM256_SET1(0.057477314f));

	DO(-0.121239071f);
	DO(0.195635925f);
	DO(-0.332994597f);
	DO(0.999995630f);

	auto res = t0 * r;

	#undef DO

	res = _mm256_blendv_ps(res, 1.570796327f - res, _mm256_cmp_ps(y_abs, x_abs, _CMP_GT_OQ));
	res = _mm256_blendv_ps(res, 3.141592654f - res, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ));
	res = _mm256_blendv_ps(res, rsign_8(res), _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_LT_OQ));

	res = _mm256_and_ps(res, _mm256_xor_ps(invalid, MM256_SET1(std::bit_cast<float>(0xFFFFFFFF))));

	return res;
}


__m256 intel_log_8(__m256 x) {

	// http://gruntthepeon.free.fr/ssemath/sse_mathfun.h

	const __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

	x = _mm256_max_ps(x, MM256_SET1(std::bit_cast<float>(0x00800000)));  /* cut off denormalized stuff */

	auto imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

	/* keep only the fractional part */
	x = _mm256_and_ps(x, MM256_SET1(std::bit_cast<float>(~0x7f800000)));
	x = _mm256_or_ps(x, MM256_SET1(0.5f));

	imm0 = _mm256_sub_epi32(imm0, _mm256_set1_epi32(0x7f));

	const auto mask = _mm256_cmp_ps(x, MM256_SET1(0.707106781186547524f), _CMP_LT_OS);

	const auto e = (1.f + _mm256_cvtepi32_ps(imm0)) - _mm256_and_ps(MM256_SET1(1.f), mask);

	x = (x - 1.f) + _mm256_and_ps(x, mask);

	const auto sqr_x = x * x;

	__m256 y = _mm256_fmadd_ps(MM256_SET1(7.0376836292E-2f), x, MM256_SET1(-1.1514610310E-1f));

	y = _mm256_fmadd_ps(y, x, MM256_SET1(1.1676998740E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(-1.2420140846E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(+1.4249322787E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(-1.6668057665E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(+2.0000714765E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(-2.4999993993E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(+3.3333331174E-1f));

	y = y * x;
	y = y * sqr_x;

	y = _mm256_fmadd_ps(e, MM256_SET1(-2.12194440e-4f), y);
	y = _mm256_fmadd_ps(sqr_x,  MM256_SET1(-0.5f), y);

	x = _mm256_fmadd_ps(e, MM256_SET1(0.693359375f), x + y);

	x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN

	return x;
}

__m256 log10_8(__m256 x) {
	return _mm256_mul_ps(x, _mm256_set1_ps(0.4342944819f)); //log(x) / In(10)
}

__m256 intel_exp_8(__m256 x) {

	//http://gruntthepeon.free.fr/ssemath/sse_mathfun.h

	x = _mm256_min_ps(x, MM256_SET1(88.3762626647949f));
	x = _mm256_max_ps(x, MM256_SET1(-88.3762626647949f));

	/* express exp(x) as exp(g + n*log(2)) */
	__m256 fx = _mm256_fmadd_ps(x, MM256_SET1(1.44269504088896341f), MM256_SET1(0.5f));

	const __m256 flr = _mm256_floor_ps(fx);

	/* if greater, subtract 1 */
	fx = flr - _mm256_and_ps(MM256_SET1(1.f), _mm256_cmp_ps(flr, fx, _CMP_GT_OS));

	x = x - _mm256_fmadd_ps(MM256_SET1(0.693359375f), fx, -2.12194440e-4f * fx);

	__m256 y = _mm256_fmadd_ps(MM256_SET1(1.9875691500E-4f), x, MM256_SET1(1.3981999507E-3f));

	y = _mm256_fmadd_ps(y, x, MM256_SET1(8.3334519073E-3f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(4.1665795894E-2f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(1.6666665459E-1f));
	y = _mm256_fmadd_ps(y, x, MM256_SET1(5.0000001201E-1f));
	y = _mm256_fmadd_ps(y, x * x, 1.f + x);

	/* build 2^n */
	const __m256i imm0 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);

	return y * _mm256_castsi256_ps(imm0);
}

_inline __m256 pow_3p5_8(const __m256 x) {

	const auto sqr = _mm256_sqrt_ps(x);

	return sqr * x * x * x;
}

_inline __m256 pow_8(const __m256& base, const __m256& power) {
	return intel_exp_8(power * intel_log_8(base));
}
