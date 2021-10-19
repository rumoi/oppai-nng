#pragma once

#include <immintrin.h>
#include <array>

#define _256_HADD(x)(\
	x.m256_f32[0] + x.m256_f32[1] + x.m256_f32[2] + x.m256_f32[3] +\
	x.m256_f32[4] + x.m256_f32[5] + x.m256_f32[6] + x.m256_f32[7])

//#define _256_HADD(x) [x]{MM_ALIGN float _v[8]; _mm256_store_ps(_v, x); return _v[0] + _v[1] + _v[2] + _v[3] + _v[4] + _v[5] + _v[6] + _v[7];}();


std::array<float, 8> to_array(__m256 x) {
	return *(std::array<float, 8>*)&x;
}

__m256 nv_sin_8(__m256 x) {

	//https://developer.download.nvidia.com/cg/sin.html

	struct float4 { float x, y, z, w; };

	constexpr float4 c0 = float4{ 0.f, 0.5f, 1.f, 0.f };
	constexpr float4 c1 = float4{ 0.25f, -9.0f, 0.75f, 0.159154943091f };
	constexpr float4 c2 = float4{ 24.9808039603f, -24.9808039603f, -60.1458091736f, 60.1458091736f };
	constexpr float4 c3 = float4{ 85.4537887573f, -85.4537887573f, -64.9393539429f, 64.9393539429f };
	constexpr float4 c4 = float4{ 19.7392082214f, -19.7392082214f, -1.0, 1.0 };

	__m256 r0_x, r0_y, r0_z;

	__m256 r1_x, r1_y, r1_z;

	__m256 r2_x{}, r2_y{}, r2_z{};

	{
		const __m256 ONE{ _mm256_set1_ps(1.f) };

		r1_x = _mm256_sub_ps(_mm256_mul_ps(x, _mm256_set1_ps(c1.w)), _mm256_set1_ps(c1.x));// only difference from cos!
		r1_y = _mm256_sub_ps(r1_x, _mm256_floor_ps(r1_x));// and extract fraction

		r2_x = _mm256_blendv_ps(r2_x, ONE, _mm256_cmp_ps(r1_y, _mm256_set1_ps(c1.x), _CMP_LT_OQ));// range check: 0.0 to 0.25
		r2_y = _mm256_blendv_ps(r2_y, ONE, _mm256_cmp_ps(r1_y, _mm256_set1_ps(c1.y), _CMP_GT_OQ));
		r2_z = _mm256_blendv_ps(r2_z, ONE, _mm256_cmp_ps(r1_y, _mm256_set1_ps(c1.z), _CMP_GT_OQ));// range check: 0.75 to 1.0

	}

	{

		const auto X{ _mm256_mul_ps(r2_x, _mm256_set1_ps(c4.z)) };
		const auto Z{ _mm256_mul_ps(r2_z, _mm256_set1_ps(c4.z)) };
		const auto Y{ _mm256_mul_ps(r2_y, _mm256_set1_ps(c4.w)) };

		r2_y = _mm256_add_ps(_mm256_add_ps(X, Z), Y);// range check: 0.25 to 0.75

	}

	r0_x = _mm256_sub_ps(_mm256_set1_ps(c0.x), r1_y);
	r0_y = _mm256_sub_ps(_mm256_set1_ps(c0.y), r1_y);
	r0_z = _mm256_sub_ps(_mm256_set1_ps(c0.z), r1_y);// range centering

	r0_x = _mm256_mul_ps(r0_x, r0_x);
	r0_y = _mm256_mul_ps(r0_y, r0_y);
	r0_z = _mm256_mul_ps(r0_z, r0_z);

	{// start power series
		r1_x = _mm256_fmadd_ps(r0_x, _mm256_set1_ps(c2.x), _mm256_set1_ps(c2.z));
		r1_y = _mm256_fmadd_ps(r0_y, _mm256_set1_ps(c2.y), _mm256_set1_ps(c2.w));
		r1_z = _mm256_fmadd_ps(r0_z, _mm256_set1_ps(c2.x), _mm256_set1_ps(c2.z));
	}

	#define DO(_x,_y,_z)\
		r1_x = _mm256_fmadd_ps(r1_x, r0_x, _mm256_set1_ps(_x));\
		r1_y = _mm256_fmadd_ps(r1_y, r0_y, _mm256_set1_ps(_y));\
		r1_z = _mm256_fmadd_ps(r1_z, r0_z, _mm256_set1_ps(_z));

	DO(c3.x, c3.y, c3.x);
	DO(c3.z, c3.w, c3.z);
	DO(c4.x, c4.y, c4.x);
	DO(c4.z, c4.w, c4.z);

	#undef DO

	const __m256 N_ONE{ _mm256_set1_ps(-1.f) };// Seems to better as a mul than a sub on MSVC, TODO: Test on clang

	const auto X{ _mm256_mul_ps(r1_x, _mm256_mul_ps(N_ONE, r2_x)) };
	const auto Y{ _mm256_mul_ps(r1_y, _mm256_mul_ps(N_ONE, r2_y)) };
	const auto Z{ _mm256_mul_ps(r1_z, _mm256_mul_ps(N_ONE, r2_z)) };

	return _mm256_add_ps(_mm256_add_ps(X, Z), Y);// range extract
}

_inline __m256 pow2_8(__m256 x) { return _mm256_mul_ps(x, x); }

__m256 dot_4_8(const std::array<std::array<float, 8>, 4>& __restrict/*must be 32 aligned*/ x) {

	const auto MAX = _mm256_set1_ps(1.f);
	const auto MIN = _mm256_set1_ps(-1.f);

	auto X1 = _mm256_load_ps(&x[0][0]);
	auto Y1 = _mm256_load_ps(&x[1][0]);
	auto X2 = _mm256_load_ps(&x[2][0]);
	auto Y2 = _mm256_load_ps(&x[3][0]);

	{// Normalize input vectors, approximated

		const auto D1 = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(X1, X1), _mm256_mul_ps(Y1, Y1)));

		X1 = _mm256_mul_ps(X1, D1);
		Y1 = _mm256_mul_ps(Y1, D1);

		const auto D2 = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(X2, X2), _mm256_mul_ps(Y2, Y2)));

		X2 = _mm256_mul_ps(X2, D2);
		Y2 = _mm256_mul_ps(Y2, D2);

	}

	auto DOT = _mm256_add_ps(_mm256_mul_ps(X1, X2), _mm256_mul_ps(Y1, Y2));

	// Clamp between -1 and 1
	DOT = _mm256_min_ps(DOT, _mm256_set1_ps(1.f));

	return _mm256_max_ps(DOT, _mm256_set1_ps(-1.f));
}

__m256 nv_acos_8(__m256 x) {

	// https://developer.download.nvidia.com/cg/acos.html

	auto negate = _mm256_set1_ps(0.f);

	{
		const auto inv = _mm256_sub_ps(negate, x);

		const auto comp = _mm256_cmp_ps(negate, x, _CMP_GT_OQ);

		negate = _mm256_blendv_ps(negate, _mm256_set1_ps(1.f), comp);

		x = _mm256_blendv_ps(x, inv, comp);
	}

	//const auto sqr = _mm256_rcp_ps(_mm256_rsqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.f), x)));
	const auto sqr = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.f), x));

	auto ret = _mm256_fmadd_ps(x, _mm256_set1_ps(-0.0187293f), _mm256_set1_ps(0.0742610f));

	ret = _mm256_fmadd_ps(x, ret, _mm256_set1_ps(-0.2121144f));
	ret = _mm256_fmadd_ps(x, ret, _mm256_set1_ps(1.5707288f));

	ret = _mm256_mul_ps(ret, sqr);

	const auto inv_negate = _mm256_mul_ps(negate, _mm256_set1_ps(-2.f));

	ret = _mm256_fmadd_ps(ret, inv_negate, ret);

	return _mm256_fmadd_ps(negate, _mm256_set1_ps(float(M_PI)), ret);
}

__m256 intel_log_8(__m256 x) {

	// http://gruntthepeon.free.fr/ssemath/sse_mathfun.h

	const __m256 one = _mm256_set1_ps(1.f);

	const __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

	x = _mm256_max_ps(x, _mm256_set1_ps(-1.f));  /* cut off denormalized stuff */

	auto imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

	/* keep only the fractional part */
	x = _mm256_and_ps(x, _mm256_set1_ps(std::bit_cast<float>(~0x7f800000)));
	x = _mm256_or_ps(x, _mm256_set1_ps(0.5f));

	imm0 = _mm256_sub_epi32(imm0, _mm256_set1_epi32(0x7f));

	__m256 e = _mm256_add_ps(_mm256_cvtepi32_ps(imm0), one);

	__m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OS);

	const __m256 tmp = _mm256_and_ps(x, mask);
	x = _mm256_sub_ps(x, one);
	e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
	x = _mm256_add_ps(x, tmp);

	const __m256 z = _mm256_mul_ps(x, x);

	__m256 y = _mm256_fmadd_ps(_mm256_set1_ps(7.0376836292E-2f), x, _mm256_set1_ps(-1.1514610310E-1f));

	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.1676998740E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.2420140846E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(+1.4249322787E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.6668057665E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(+2.0000714765E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-2.4999993993E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(+3.3333331174E-1f));

	y = _mm256_mul_ps(y, x);
	y = _mm256_mul_ps(y, z);

	y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.12194440e-4f), y);

	y = _mm256_fmadd_ps(z, _mm256_set1_ps(-0.5f), y);

	x = _mm256_add_ps(x, y);
	x = _mm256_fmadd_ps(e, _mm256_set1_ps(0.693359375f), x);
	x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN

	return x;
}

__m256 log10_8(__m256 x) {
	return _mm256_mul_ps(x, _mm256_set1_ps(0.4342944819f)); //log(x) / In(10)
}

__m256 intel_exp_8(__m256 x) {

	//http://gruntthepeon.free.fr/ssemath/sse_mathfun.h

	const __m256 one = _mm256_set1_ps(1.f);

	x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
	x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

	/* express exp(x) as exp(g + n*log(2)) */
	__m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));

	__m256 tmp = _mm256_floor_ps(fx);

	/* if greater, subtract 1 */
	__m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
	mask = _mm256_and_ps(mask, one);
	fx = _mm256_sub_ps(tmp, mask);

	tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f));

	__m256 z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4));

	x = _mm256_sub_ps(x, tmp);
	x = _mm256_sub_ps(x, z);

	z = _mm256_mul_ps(x, x);

	__m256 y = _mm256_fmadd_ps(_mm256_set1_ps(1.9875691500E-4f), x, _mm256_set1_ps(1.3981999507E-3f));

	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
	y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
	y = _mm256_fmadd_ps(y, z, x);

	y = _mm256_add_ps(y, one);

	/* build 2^n */
	__m256i imm0 = _mm256_cvttps_epi32(fx);

	imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
	imm0 = _mm256_slli_epi32(imm0, 23);

	return _mm256_mul_ps(y, _mm256_castsi256_ps(imm0));
}

_inline __m256 pow_3p5_8(const __m256 x) {

	const auto sqr = _mm256_sqrt_ps(x);

	return _mm256_mul_ps(_mm256_mul_ps(x, sqr), pow2_8(x));
}

_inline __m256 pow_8(const __m256& __restrict base, const __m256& __restrict power) {
	return intel_exp_8(_mm256_mul_ps(power, intel_log_8(base)));
}
