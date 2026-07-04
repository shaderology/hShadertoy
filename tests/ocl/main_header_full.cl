/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    interpolate.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 */

#ifndef __INTERPOLATE_H__
#define __INTERPOLATE_H__

// #pragma OPENCL EXTENSION cl_amd_printf : enable

/*#ifdef cl_amd_printf
#define CHECKNAN(f) if (isnan(f)) { printf("%s is NAN, x = %g, y = %g, z = %g\n",#f, x, y, z);}
#define CHECKNAN2D(f) if (isnan(f)) { printf("%s is NAN, x = %g, y = %g\n",#f, x, y);}
#else*/
#define CHECKNAN(f)
#define CHECKNAN2D(f)
// #endif

#define CLAMPOFFSET 1e-4f

// While mix is limited to 0..1, it is tempting to just use a clamp
// on t; but you can get cancellation at t == 1 that stops the result
// from being b.   Selecting on t == 0 will also stop any NANs from
// leaking across, which matches the theory that a mix with 0 should
// able to be short-circuited with an if (t == 0) a, so the fact the
// second component is nan should not affect the result in this case.
#define _SAFE_MIX(a, b, t) \
    select( select( mix(a, b, t), a, t <= 0 ), b, t >= 1 )

static float safe_mix1(float a, float b, float t) { return _SAFE_MIX(a, b, t); }
static float2 safe_mix2(float2 a, float2 b, float2 t) { return _SAFE_MIX(a, b, t); }
static float3 safe_mix3(float3 a, float3 b, float3 t) { return _SAFE_MIX(a, b, t); }
static float4 safe_mix4(float4 a, float4 b, float4 t) { return _SAFE_MIX(a, b, t); }

#undef _SAFE_MIX

// The build methods are usually inferior to the direct cast,
// but ensure the result is an RValue so people can't assign to
// them in an unexpected manner.
static float2
build_float2(float x, float y)
{
    return (float2)(x, y);
}
static float3
build_float3(float x, float y, float z)
{
    return (float3)(x, y, z);
}
static float4
build_float4(float x, float y, float z, float w)
{
    return (float4)(x, y, z, w);
}

static float
lerp(float v1, float v2, float t)
{
    return v1 + (v2 - v1)*t;
}

static float
fit01(float t, float v1, float v2)
{
    return mix(v1, v2, clamp(t, 0.0f, 1.0f));
}

static float
fitTo01(float val, float omin, float omax)
{
    float d = omax - omin;
    if (fabs(d) < 1e-8f)
        return 0.5f;
    if (omin < omax)
    {
        if (val < omin) return 0;
        if (val > omax) return 1;
    }
    else
    {
        if (val < omax) return 1;
        if (val > omin) return 0;
    }
    return (val - omin) / d;
}

static float
fit(float val, float omin, float omax, float nmin, float nmax)
{
    return mix(nmin, nmax, fitTo01(val, omin, omax));
}

// in is an array size long of values to be interpolated within the
// [0, 1] interval.
// pos==0 => in[0]
// pos==1 => in[size-1]
static float
lerpConstant( constant float * in, int size, float pos )
{
    int m = size - 1;
    float flr;
    float t = fract(clamp(pos, 0.0f, 1.0f) * m, &flr);
    int flooridx = convert_int(flr);
    int ceilidx = min(flooridx+1, m);
    return mix(in[flooridx], in[ceilidx], t);
}

static float3
lerpConstant3( constant float * in, int size, float pos )
{
    int m = size - 1;
    float flr;
    float t = fract(clamp(pos, 0.0f, 1.0f) * m, &flr);
    int flooridx = convert_int(flr);
    int ceilidx = min(flooridx+1, m);
    float3 v1 = vload3(flooridx, in);
    float3 v2 = vload3(ceilidx, in);
    return mix(v1, v2, t);
}

static float
centerFromFace(__global const float *a, size_t idx, uint axisstride)
{
    return 0.5f * (a[idx] + a[idx + axisstride]);
}

static float
faceFromCenter(__global const float *a, size_t idx, uint axisstride)
{
    return 0.5f * (a[idx - axisstride] + a[idx]);
}

static float
cornerFromCenter(__global const float *a, size_t idx,
                 uint ystride, uint zstride)
{
    return 0.125f * (a[idx]
                   + a[idx - 1]
                   + a[idx - ystride]
                   + a[idx - zstride]
                   + a[idx - 1 - ystride]
                   + a[idx - 1 - zstride]
                   + a[idx - ystride - zstride]
                   + a[idx - 1 - ystride - zstride]);
}
static float
cornerFromCenter2d(__global const float *a, size_t idx,
                   uint xstride, uint ystride)
{
    return 0.25f * (a[idx] + a[idx - xstride] + a[idx - ystride]
                           + a[idx - xstride - ystride]);
}

// Calc central difference derivative of cell-centered grid
// at a center cell at idx.
static float
dudxAligned(__global const float *u, const uint idx,
                  const uint xstride, const float inv2dx)
{
    return inv2dx * (u[idx + xstride] - u[idx - xstride]);
}

// Calc central difference derivative of face-sampled grid
// at a center cell at idx.  This works for all the off-axes directions
// other than the one the face-sampled grid represents.  E.g. you can
// take the dy-derivative of the x-velocity field.
static float
dudxAlignedFace(__global const float *u, const uint idx,
                      const uint ustride, const uint xstride, float inv4dx)
{
    return inv4dx * ((u[idx + xstride] + u[idx + ustride + xstride]) -
                     (u[idx - xstride] + u[idx + ustride - xstride]));
}

// Calc central difference derivative of face-sampled grid
// at a center cell at idx. This only works for the derivative along
// the axis the face-centered grid represents, e.g. you can only take
// the dx-derivative of the x-velocity field.
static float
dudxFaceAtCenter(__global const float *u, const uint idx,
                      const uint xstride, float invdx)
{
    return invdx * (u[idx + xstride] - u[idx]);
}

static float
dudxCenterAtFace(__global const float *u, const uint idx,
                      const uint xstride, float invdx)
{
    return invdx * (u[idx] - u[idx - xstride]);
}

// Calc central difference derivative of center-sampled grid at a corner cell
// at idx, by first averaging along relevant faces (invdx should incorporate
// the averaging factor). axis_stride is stride along the differentiation axis,
// and the other strides are along the remaining axes.
static float
dudxCenterAtCorner(__global const float *u, const uint idx,
                   const uint axis_stride, const uint off_stride1,
                   const uint off_stride2, float inv4dx)
{
    return inv4dx * ((u[idx] + u[idx - off_stride1]
            + u[idx - off_stride2] + u[idx - off_stride1 - off_stride2])
                 -  (u[idx - axis_stride] + u[idx - off_stride1 - axis_stride]
            + u[idx - off_stride2 - axis_stride]
            + u[idx - off_stride1 - off_stride2 - axis_stride]));
}

static float
dudxCenterAtCorner2d(__global const float *u, const uint idx,
                     const uint axis_stride, const uint off_stride,
                     float inv2dx)
{
    return inv2dx * ((u[idx] + u[idx - off_stride])
                 -  (u[idx - axis_stride] + u[idx - off_stride - axis_stride]));
}

static void
bilinear_interp(float x, float y, __global const float *p,
                        size_t idx,
                        __global float *phin,
                        __global float *minphi,
                        __global float *maxphi,
                        uint offset, uint xstride, uint ystride)

{
    // clamp to boundaries
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);

    // get fractional part
    const float sx = x - gi;
    const float sy = y - gj;

    size_t srcidx = offset + gi * xstride + gj * ystride;

    const float i00 = p[srcidx];
    const float i10 = p[srcidx + xstride];
    const float i01 = p[srcidx + ystride];
    const float i11 = p[srcidx + xstride + ystride];

    CHECKNAN2D(i00)
    CHECKNAN2D(i10)
    CHECKNAN2D(i01)
    CHECKNAN2D(i11)
    const float val = (i00 * (1-sx) + i10 * (sx)) * (1-sy) +
                      (i01 * (1-sx) + i11 * (sx)) * (  sy);
    phin[idx] = val;
    if (minphi)
        minphi[idx] = fmin(fmin(fmin(i00, i01), i10), i11);
    if (maxphi)
        maxphi[idx] = fmax(fmax(fmax(i00, i01), i10), i11);
}

static float
bilinear_interp_val(float x, float y, __global const float *p,
                        uint offset, uint xstride, uint ystride,
                        uint offx,  uint offy)
{
    // clamp to boundaries
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET - offx);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET - offy);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);

    // get fractional part
    const float sx = x - gi;
    const float sy = y - gj;

    size_t srcidx = offset + gi * xstride + gj * ystride;

    const float i00 = p[srcidx];
    const float i10 = p[srcidx + xstride];
    const float i01 = p[srcidx + ystride];
    const float i11 = p[srcidx + xstride + ystride];

    return (i00 * (1-sx) + i10 * (sx)) * (1-sy) +
           (i01 * (1-sx) + i11 * (sx)) * (  sy);
}

static void
trilinear_interp(float x, float y, float z, __global const float *p,
                    size_t idx,
                    __global float *phin,
                    __global float *minphi,
                    __global float *maxphi,
                    uint offset, uint ystride, uint zstride)
{
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET);
    z = clamp(z, -1.0f, get_global_size(2) - CLAMPOFFSET);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);
    const int gk = (int)floor(z);

    const float sx = x - gi;
    const float sy = y - gj;
    const float sz = z - gk;

    size_t srcidx = offset + gi + gj * ystride + gk * zstride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + 1];
    const float i010 = p[srcidx + ystride];
    const float i110 = p[srcidx + 1 + ystride];
    const float i001 = p[srcidx + zstride];
    const float i101 = p[srcidx + 1 + zstride];
    const float i011 = p[srcidx + ystride + zstride];
    const float i111 = p[srcidx + 1 + ystride + zstride];

    CHECKNAN(i000)
    CHECKNAN(i100)
    CHECKNAN(i010)
    CHECKNAN(i110)
    CHECKNAN(i001)
    CHECKNAN(i101)
    CHECKNAN(i011)
    CHECKNAN(i111)
    const float val = ((i000 * (1 - sx) + i100 * (sx)) * (1 - sy) +
                      (i010 * (1 - sx) + i110 * (sx)) * (  sy)) * (1 - sz) +
                      ((i001 * (1 - sx) + i101 * (sx)) * (1 - sy) +
                      (i011 * (1 - sx) + i111 * (sx)) * (  sy)) * (  sz);

    phin[idx] = val;
    if (minphi)
        minphi[idx] = fmin(fmin(fmin(fmin(fmin(fmin(fmin(i000, i001), i010),
                                            i011), i100), i101), i110), i111);
    if (maxphi)
        maxphi[idx] = fmax(fmax(fmax(fmax(fmax(fmax(fmax(i000, i001), i010),
                                            i011), i100), i101), i110), i111);
}

static float
trilinear_interp_val(float x, float y, float z, __global const float *p,
                    uint offset, uint ystride, uint zstride,
                    uint offx,  uint offy, uint offz )
{
    x = clamp(x, -1.0f, get_global_size(0) - CLAMPOFFSET - offx);
    y = clamp(y, -1.0f, get_global_size(1) - CLAMPOFFSET - offy);
    z = clamp(z, -1.0f, get_global_size(2) - CLAMPOFFSET - offz);

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);
    const int gk = (int)floor(z);

    const float sx = x - gi;
    const float sy = y - gj;
    const float sz = z - gk;

    size_t srcidx = offset + gi + gj * ystride + gk * zstride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + 1];
    const float i010 = p[srcidx + ystride];
    const float i110 = p[srcidx + 1 + ystride];
    const float i001 = p[srcidx + zstride];
    const float i101 = p[srcidx + 1 + zstride];
    const float i011 = p[srcidx + ystride + zstride];
    const float i111 = p[srcidx + 1 + ystride + zstride];

    return  ((i000 * (1 - sx) + i100 * (sx)) * (1 - sy) +
            (i010 * (1 - sx) + i110 * (sx)) * (  sy)) * (1 - sz) +
            ((i001 * (1 - sx) + i101 * (sx)) * (1 - sy) +
            (i011 * (1 - sx) + i111 * (sx)) * (  sy)) * (  sz);
}

static float
bilinear_interp_vol(float2 pos, __global const float *p,
                    uint offset, uint xstride, uint ystride,
                    uint resx, uint resy)
{
    float x = clamp(pos.x-0.5f, (float)0, (float)(resx-1));
    float y = clamp(pos.y-0.5f, (float)0, (float)(resy-1));

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);

    // In case our clamp is exactly res#-1
    const int dx = select((int)xstride, (int)0, gi == (int)(resx)-1);
    const int dy = select((int)ystride, (int)0, gj == (int)(resy)-1);

    const float sx = x - gi;
    const float sy = y - gj;

    size_t srcidx = offset + gi * xstride + gj * ystride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + dx];
    const float i010 = p[srcidx + dy];
    const float i110 = p[srcidx + dx + dy];

    return mix( mix(i000, i100, sx),
                mix(i010, i110, sx), sy);
}

static float
trilinear_interp_vol(float3 pos, __global const float *p,
                    uint offset, uint xstride, uint ystride, uint zstride,
                    uint resx, uint resy, uint resz)
{
    float x = clamp(pos.x-0.5f, (float)0, (float)(resx-1));
    float y = clamp(pos.y-0.5f, (float)0, (float)(resy-1));
    float z = clamp(pos.z-0.5f, (float)0, (float)(resz-1));

    const int gi = (int)floor(x);
    const int gj = (int)floor(y);
    const int gk = (int)floor(z);

    // In case our clamp is exactly res#-1
    const int dx = select((int)xstride, (int)0, gi == (int)(resx)-1);
    const int dy = select((int)ystride, (int)0, gj == (int)(resy)-1);
    const int dz = select((int)zstride, (int)0, gk == (int)(resz)-1);

    const float sx = x - gi;
    const float sy = y - gj;
    const float sz = z - gk;

    size_t srcidx = offset + gi * xstride + gj * ystride + gk * zstride;

    const float i000 = p[srcidx];
    const float i100 = p[srcidx + dx];
    const float i010 = p[srcidx + dy];
    const float i110 = p[srcidx + dx + dy];
    const float i001 = p[srcidx + dz];
    const float i101 = p[srcidx + dx + dz];
    const float i011 = p[srcidx + dy + dz];
    const float i111 = p[srcidx + dx + dy + dz];

    return mix( mix( mix(i000, i100, sx),
                     mix(i010, i110, sx), sy),
                mix( mix(i001, i101, sx),
                     mix(i011, i111, sx), sy),
                sz);
}

static float4
linesegInterpolationWeights(float u)
{
    return (float4)(1 - u, u, 0, 0);
}

// From GEOtriInterpolationWeights, but
static float4
triInterpolationWeights(float u, float v)
{
    // Triangle - use barycentric coordinates

    // This is a hack to make sure we are given proper
    // barycentric u, v coordinates.  That is, we require
    // u+v <= 1, and if that's not the case we hack it so
    // u = 1-u, v = 1-v, thus ensuring u+v <= 1.  (This
    // assumes that u, v are each between 0 and 1)
    // This is used for when evaluateInteriorPoint is
    // called from POP_GenVar for birthing from a surface.
    //
    // Note we actually flip on the u+v = 1 axis instead
    // of what is described above so slightly outside points
    // do not teleport to opposite locations.
    float uv = 1 - u - v;

    // Assume valid uv's.
#if 0
    if (uv < 0)
    {
        u += uv;
        v += uv;
        uv = -uv;
    }
#endif

    return (float4) (uv, u, v, 0);
}

// From GEOquadInterpolationWeights
static float4
quadInterpolationWeights(float u, float v)
{
    float u1 = 1 - u;
    float v1 = 1 - v;
    return (float4)(u1 * v1, u1 * v, u * v, u * v1);
}

// From GEO_PrimTetrahedron::remapTetCoords
static float4
tetInterpolationWeights(float u, float v, float w)
{
    float uvw = 1 - u - v - w;

    // Assume valid uv's.
#if 0
    if (uvw < 0)
    {
        // Mirror in u + v == 1 plane, reducing from 6 tetrahedra to 3,
        // i.e. a right triangular prism, whose right-angle edge is
        // along the w axis.
        if (u + v > 1)
        {
            float bary = 1 - u - v;
            u += bary;
            v += bary;
            uvw -= 2 * bary;
        }
        // Mirror the far tetrahedron (shares no face with the tet to keep)
        // into the middle tetrahedron (shares one face with the tet to keep)
        if (u + w > 1)
        {
            // Weight of point at (1,0,1), since 1+1-1 = 1
            float weight = (u + w - 1);
            // Subtract component of (1,0,1), which only changes u and w
            u -= weight;//*1
            //v -= weight*0;
            w -= weight;//*1
            // Add component of (0,1,0), which only changes v
            //u += weight*0;
            v += weight;//*1
            //w += weight*0;
            // Update uvw
            uvw += weight;
        }
        // Mirror the remaining outside tetrahedron into the tet to keep
        if (uvw < 0)
        {
            // Weight of point at (0,1,1), since -(1-0-1-1) = 1
            float weight = -uvw;
            // Subtract component of (0,1,1), which only changes v and w
            //u -= weight*0
            v -= weight;//*1
            w -= weight;//*1
            // Add component of (0,0,0), which requires no change
            // Update uvw
            uvw = -uvw; //equivalent to uvw += 2*weight;
        }
    }
#endif

    return (float4)(uvw, u, v, w);
}

static void
computeSubdCurveCoeffsAndIndices(
    float u,
    int n,
    bool closed,
    float4 *coeffs_ptr,
    int4 *indices_ptr)
{
    const float16 theSubDFirstBasis = {
        1.0, -1.0,  0.0,  1.0/6.0,
        0.0,  1.0,  0.0, -1.0/3.0,
        0.0,  0.0,  0.0,  1.0/6.0,
        0.0,  0.0,  0.0,  0.0
    };

    const float16 theOpenBasis = {
        1.0/6.0, -0.5,  0.5, -1.0/6.0,
        2.0/3.0,  0.0, -1.0,  0.5,
        1.0/6.0,  0.5,  0.5, -0.5,
        0.0,      0.0,  0.0,  1.0/6.0
    };

    float4 coeffs = 0.0;
    int4 indices = 0.0;

    // UT_ASSERT_P(n >= 1);

    // Special cases for n <= 2
    if (n == 1)
    {
        indices.x = -1;
        indices.y = 0;
        indices.z = -1;
        indices.w = -1;

        coeffs.x = 0;
        coeffs.y = 1;
        coeffs.z = 0;
        coeffs.w = 0;

        *coeffs_ptr = coeffs;
        *indices_ptr = indices;
    }
    else if (n == 2)
    {
        if (closed)
        {
            u *= 2;
            if (u > 1)
                u = 2-u;
        }

        indices.x = -1;
        indices.y = 0;
        indices.z = 1;
        indices.w = -1;

        coeffs.x = 0.0;
        coeffs.y = 1.0 - u;
        coeffs.z = u;
        coeffs.w = 0.0;
    }

    const float16 *basis = &theOpenBasis;
    // const float16 *midbasis = &theOpenBasis;

    int nedges = n - !closed;
    u *= nedges;
    int i = floor(u);
    i = clamp(i, 0, nedges-1);
    float t = u - i;

    float16 temp;
    if (i == 0)
    {
        if (closed)
        {
            indices.x = n-1;
            indices.y = 0;
            indices.z = 1;
            indices.w = 2;
        }
        else
        {
            basis = &theSubDFirstBasis;

            indices.x = 0;
            indices.y = 1;
            indices.z = 2;
            indices.w = -1;
        }
    }
    else if (i >= n-2)
    {
        if (closed)
        {
            if (i == n-2)
            {
                indices.x = n-3;
                indices.y = n-2;
                indices.z = n-1;
                indices.w = 0;
            }
            else
            {
                // UT_ASSERT_P(i == n-1);
                indices.x = n-2;
                indices.y = n-1;
                indices.z = 0;
                indices.w = 1;
            }
        }
        else
        {
            basis = &theSubDFirstBasis;

            indices.x = -1;
            indices.y = n-3;
            indices.z = n-2;
            indices.w = n-1;

            temp.lo.lo = (*basis).hi.hi;
            temp.lo.hi = (*basis).hi.lo;
            temp.hi.lo = (*basis).lo.hi;
            temp.hi.hi = (*basis).lo.lo;
            basis = &temp;
            t = 1-t;
        }
    }
    else
    {
        indices.x = i-1;
        indices.y = i;
        indices.z = i+1;
        indices.w = i+2;
    }

    float t2 = t*t;
    float4 tpow = {1.0f, t, t2, t2*t};

    (*coeffs_ptr).x = dot(tpow, (*basis).lo.lo);
    (*coeffs_ptr).y = dot(tpow, (*basis).lo.hi);
    (*coeffs_ptr).z = dot(tpow, (*basis).hi.lo);
    (*coeffs_ptr).w = dot(tpow, (*basis).hi.hi);

    *indices_ptr = indices;
}

static float3 
evaluateCubicCoeffs_vload3(float4 coeffs, int4 indices, global float *values)
{
    float3 res = 0;

    if (indices.x >= 0)
        res += coeffs.x * vload3(indices.x, values);
    if (indices.y >= 0)
        res += coeffs.y * vload3(indices.y, values);
    if (indices.z >= 0)
        res += coeffs.z * vload3(indices.z, values);
    if (indices.w >= 0)
        res += coeffs.w * vload3(indices.w, values);

    return res;
}

static float3 
evaluateCubicCoeffs_3(float4 coeffs, int4 indices, float3 *values)
{
    float3 res = 0;

    if (indices.x >= 0)
        res += coeffs.x * values[indices.x];
    if (indices.y >= 0)
        res += coeffs.y * values[indices.y];
    if (indices.z >= 0)
        res += coeffs.z * values[indices.z];
    if (indices.w >= 0)
        res += coeffs.w * values[indices.w];

    return res;
}
#endif
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    typedefines.h ( CE Library, OpenCL)
 *
 * COMMENTS: OpenCL type definitions
 */


#ifndef __TYPE_DEFINE_H__
#define __TYPE_DEFINE_H__

// The OpenCL SOP/DOP might define
// these before compilation for 32-/64-bit support,
// so only define them if not already defined.
#ifndef fpreal
// The USE_DOUBLE flag is used by the older OpenCL-enabled
// DOPs that make up the Pyro solver.
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define fpreal double
#define fpreal2 double2
#define fpreal3 double3
#define fpreal4 double4
#define fpreal8 double8
#define fpreal16 double16
#define FPREAL_PREC 64
// We also want to define exint as long in this case.
#define USE_LONG

#else
#define fpreal float
#define fpreal2 float2
#define fpreal3 float3
#define fpreal4 float4
#define fpreal8 float8
#define fpreal16 float16
#define FPREAL_PREC 32

#endif
#endif


#if FPREAL_PREC==64

// Load a 64-bit fpreal2 from a float2 buffer.
static fpreal2
vload2f(size_t i, const global float *b)
{
    i *= 2;
    return (fpreal2)(b[i], b[i + 1]);
}

// Load a 64-bit fpreal3 from a float3 buffer.
static fpreal3
vload3f(size_t i, const global float *b)
{
    i *= 3;
    return (fpreal3)(b[i], b[i+1], b[i+2]);
}

// Load a 64-bit fpreal4 from a float4 buffer.
static fpreal4
vload4f(size_t i, const global float *b)
{
    i *= 4;
    return (fpreal4)(b[i], b[i+1], b[i+2], b[i+3]);
}

// Store a 64-bit fpreal3 into a float3 buffer.
static void
vstore3f(fpreal3 a, size_t i, global float *b)
{
    vstore3((float3)(a.x, a.y, a.z), i, b);
}

// Store a 64-bit fpreal4 into a float4 buffer.
static void
vstore4f(fpreal4 a, size_t i, global float *b)
{
    vstore4((float4)(a.x, a.y, a.z, a.w), i, b);
}

// Convert float2 to 64-bit fpreal2
static fpreal2
asfpreal2(float2 a)
{
    return (fpreal2)(a.x, a.y);
}

// Convert float3 to 64-bit fpreal3
static fpreal3
asfpreal3(float3 a)
{
    return (fpreal3)(a.x, a.y, a.z);
}

// Convert float4 to 64-bit fpreal4
static fpreal4
asfpreal4(float4 a)
{
    return (fpreal4)(a.x, a.y, a.z, a.w);
}

#else

// Load a 32-bit fpreal2 from a float2 buffer (no-op).
#define vload2f vload2

// Load a 32-bit fpreal3 from a float3 buffer (no-op).
#define vload3f vload3

// Load a 32-bit fpreal4 from a float4 buffer (no-op).
#define vload4f vload4

// Store a 32-bit fpreal3 into a float3 buffer (no-op).
#define vstore3f vstore3

// Store a 32-bit fpreal4 into a float4 buffer (no-op).
#define vstore4f vstore4

// Convert float2 to 32-bit fpreal2 (no-op)
#define asfpreal2(x) (x)

// Convert float3 to 32-bit fpreal3 (no-op)
#define asfpreal3(x) (x)

// Convert float4 to 32-bit fpreal4 (no-op)
#define asfpreal4(x) (x)

#endif

// The OpenCL SOP/DOP might define
// these before compilation for 32-/64-bit support,
// so only define them if not already defined.
#ifndef exint
#ifdef USE_LONG
#define exint  long
#define exint2 long2
#define exint3 long3
#define exint4 long4
#else
#define exint  int
#define exint2 int2
#define exint3 int3
#define exint4 int4
#endif
#endif

#endif
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    util.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 */

#ifndef __UTIL_H__
#define __UTIL_H__

static void swapf(fpreal *a, fpreal *b)
{
    fpreal t = *a;
    *a = *b;
    *b = t;
}

#endif
/*
 * PROPRIETARY INFORMATION.  This software is proprietary to
 * Side Effects Software Inc., and is not to be reproduced,
 * transmitted, or disclosed in any way without written permission.
 *
 * Produced by:
 *  Side Effects Software Inc
 *  123 Front Street West, Suite 1401
 *  Toronto, Ontario
 *  Canada   M5J 2M2
 *  416-504-9876
 *
 * NAME:    matrix.h ( CE Library, OpenCL)
 *
 * COMMENTS:
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

// #include "typedefines.h"
// #include "util.h"

#define PRINTI(v)                                                              \
    printf("%s:\n", #v);                                                       \
    printf("%d\n", v)

#define PRINTU(v)                                                              \
    printf("%s:\n", #v);                                                       \
    printf("%u\n", v)

#define PRINTLI(v)                                                             \
    printf("%s:\n", #v);                                                       \
    printf("%lld\n", v)

#define PRINTLU(v)                                                             \
    printf("%s:\n", #v);                                                       \
    printf("%llu\n", v)

#define PRINTF(v)                                                              \
    printf("%s:\n", #v);                                                       \
    printf("%g\n", v)

#define PRINTVEC3(v)                                                           \
    printf("%s:\n", #v);                                                       \
    printf("%g %g %g\n", v.x, v.y, v.z)

#define PRINTVEC3I(v)                                                          \
    printf("%s:\n", #v);                                                       \
    printf("%d %d %d\n", v.x, v.y, v.z)

#define PRINTMAT3(m)                                                           \
    printf("%s:\n", #m);                                                       \
    printf("%g %g %g\n", m[0].s0, m[0].s1, m[0].s2);                           \
    printf("%g %g %g\n", m[1].s0, m[1].s1, m[1].s2);                           \
    printf("%g %g %g\n", m[2].s0, m[2].s1, m[2].s2)

#define PRINTMAT3NP(m)                                                         \
    printf("%s = np.array((\n\t(%f, %f, %f),\n\t(%f, %f, %f),\n\t(%f, %f, %f)))\n", \
            #m, m[0].s0, m[0].s1, m[0].s2, m[1].s0, m[1].s1, m[1].s2, m[2].s0, m[2].s1, m[2].s2);

#define PRINTMAT4(m)                                                           \
    printf("%s:\n", #m);                                                       \
    printf("%g %g %g %g\n", m.s0, m.s1, m.s2, m.s3);                           \
    printf("%g %g %g %g\n", m.s4, m.s5, m.s6, m.s7);                           \
    printf("%g %g %g %g\n", m.s8, m.s9, m.sa, m.sb);                           \
    printf("%g %g %g %g\n", m.sc, m.sd, m.se, m.sf)

// A 3x3 matrix in row-major order (to match UT_Matrix3)
// NOTE: fpreal3 is 4 floats, so this is size 12
typedef fpreal3 mat3[3];  

// A 3x2 matrix in row-major order
typedef fpreal2 mat32[3];

// A 2x2 matrix in row-major order, stored in a single fpreal4
typedef fpreal4 mat2;

// A 4x4 matrix in row-major order, stored in a single fpreal16
typedef fpreal16 mat4;

// Return the sum of entries of the vector v.
static fpreal
vec3sum(const fpreal3 v)
{
    return v.x + v.y + v.z;
}

// Return the product of entries of the vector v.
static fpreal
vec3prod(const fpreal3 v)
{
    return v.x * v.y * v.z;
}

// Create a 2x2 matrix with columns of the specified vectors.
static mat2
mat2fromcols(const fpreal2 c0, const fpreal2 c1)
{
    return (mat2)(c0.s0, c1.s0,
                  c0.s1, c1.s1);
}

// Transpose a 2x2 matrix.
static mat2
transpose2(const mat2 a)
{
    return (mat2)(a.even, a.odd);
}

// Multiply A * B for 2x2 matrices.
static mat2
mat2mul(const mat2 a, const mat2 b)
{
    return (mat2)(dot(a.lo, b.even), dot(a.lo, b.odd),
                  dot(a.hi, b.even), dot(a.hi, b.odd));
}

// Multiply b * A where b is a 2-vector and A is a 2x2 matrix.
// This multiplication order matches VEX.
static fpreal2
mat2vecmul(const mat2 a, const fpreal2 b)
{
    mat2 aT = transpose2(a);
    return (fpreal2)(dot(aT.lo, b), dot(aT.hi, b));
}

// Return the square of the L2-norm of a 2x2 matrix.
static fpreal
squaredNorm2(const mat2 a)
{
    return dot(a.lo, a.lo) + dot(a.hi, a.hi);
}

// Add 3x3 matrix A to matrix B and store the result in matrix C.
static void
mat3add(const mat3 a, const mat3 b, mat3 c)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}

// Subtract 3x3 matrix A from matrix B and store the result in C.
static void
mat3sub(const mat3 a, const mat3 b, mat3 c)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

// Set 3x3 matrix A to zero.
static void
mat3zero(mat3 a)
{
    a[0] = (fpreal3)(0.0f);
    a[1] = (fpreal3)(0.0f);
    a[2] = (fpreal3)(0.0f);
}

// Set 3x3 matrix A to the identity.
static void
mat3identity(mat3 a)
{
    a[0] = (fpreal3)(1.0f, 0.0f, 0.0f);
    a[1] = (fpreal3)(0.0f, 1.0f, 0.0f);
    a[2] = (fpreal3)(0.0f, 0.0f, 1.0f);
}

// Copy 3x3 matrix A to matrix B.
static void
mat3copy(const mat3 a, mat3 b)
{
    b[0] = a[0];
    b[1] = a[1];
    b[2] = a[2];
}

// Load a 3x3 matrix from memory at the specified index.
// The matrix should be row-major as stored in geometry
// attributes.
static void
mat3load(size_t idx, const global float *a, mat3 m)
{
    idx *= 3;
    m[0] = vload3f(idx, a);
    m[1] = vload3f(idx + 1, a);
    m[2] = vload3f(idx + 2, a);
}

// Store a 3x3 matrix to memory at the specified index.
// The matrix should be row-major as stored in geometry
// attributes.
static void 
mat3store(mat3 in, int idx, global fpreal *data)
{
    idx *= 3;
    vstore3(in[0], idx, data);
    vstore3(in[1], idx + 1, data);
    vstore3(in[2], idx + 2, data);
}

// Create a 3x3 matrix with columns of the specified vectors.
static void
mat3fromcols(const fpreal3 c0, const fpreal3 c1, const fpreal3 c2, mat3 m)
{
    m[0] = (fpreal3)(c0.s0, c1.s0, c2.s0);
    m[1] = (fpreal3)(c0.s1, c1.s1, c2.s1);
    m[2] = (fpreal3)(c0.s2, c1.s2, c2.s2);
}

// Transpose a 3x3 matrix.
static void
transpose3(const mat3 a, mat3 b)
{
    mat3fromcols(a[0], a[1], a[2], b);
}

// Multiply A * B and store in C for 3x3 matrices.
static void
mat3mul(const mat3 a, const mat3 b, mat3 c)
{
    mat3 bT;
    transpose3(b, bT);
    c[0] = (fpreal3)(dot(a[0], bT[0]), dot(a[0], bT[1]), dot(a[0], bT[2]));
    c[1] = (fpreal3)(dot(a[1], bT[0]), dot(a[1], bT[1]), dot(a[1], bT[2]));
    c[2] = (fpreal3)(dot(a[2], bT[0]), dot(a[2], bT[1]), dot(a[2], bT[2]));
}

// Multiply A * B^T and store in C for 3x3 matrices.
static void
mat3mulT(const mat3 a, const mat3 b, mat3 c)
{
    c[0] = (fpreal3)(dot(a[0], b[0]), dot(a[0], b[1]), dot(a[0], b[2]));
    c[1] = (fpreal3)(dot(a[1], b[0]), dot(a[1], b[1]), dot(a[1], b[2]));
    c[2] = (fpreal3)(dot(a[2], b[0]), dot(a[2], b[1]), dot(a[2], b[2]));
}

// Multiply b * A where b is a 3-vector and A is a 3x3 matrix.
// This multiplication order matches VEX.
static fpreal3
mat3vecmul(const mat3 a, const fpreal3 b)
{
    mat3 aT;
    transpose3(a, aT);
    return (fpreal3)(dot(aT[0], b), dot(aT[1], b), dot(aT[2], b));
}

// Multiply b * A^T where b is a 3-vector and A is a 3x3 matrix.
// This multiplication order matches VEX.
static fpreal3
mat3Tvecmul(const mat3 a, const fpreal3 b)
{
    return (fpreal3)(dot(a[0], b), dot(a[1], b), dot(a[2], b));
}

// Multiply b * A where b is a 3-vector and A is a 3x3 matrix,
// but discard the third component.
// This multiplication order matches VEX.
static fpreal2
mat3vec2mul(const mat3 a, const fpreal3 b)
{
    mat3 aT;
    transpose3(a, aT);
    return (fpreal2)(dot(aT[0], b), dot(aT[1], b));
}

// Multiply b * A^T where b is a 3-vector and A is a 3x3 matrix,
// but discard the third component.
// This multiplication order matches VEX.
static fpreal2
mat3Tvec2mul(const mat3 a, const fpreal3 b)
{
    return (fpreal2)(dot(a[0], b), dot(a[1], b));
}

// Store the 3x3 matrix that is the outer product of the input
// a and b vectors in C.
static void
outerprod3(const fpreal3 a, const fpreal3 b, mat3 c)
{
    c[0] = a.x * b;
    c[1] = a.y * b;
    c[2] = a.z * b;
}

// Compute C = s * A + t * B, where s, t are scalars and A, B are 3x3 matrices.
static void
mat3lcombine(const fpreal s, const mat3 a, const fpreal t, const mat3 b, mat3 c)
{
    c[0] = s * a[0] + t * b[0];
    c[1] = s * a[1] + t * b[1];
    c[2] = s * a[2] + t * b[2];
}

// Return the square of the L2-norm of a 3x3 matrix.
static fpreal
squaredNorm3(const mat3 a)
{
    return dot(a[0], a[0]) + dot(a[1], a[1]) + dot(a[2], a[2]);
}

// Return the determinant of the supplied 3x3 matrix.
static fpreal
det3(const mat3 a)
{
    fpreal d = a[0].s0 * (a[1].s1 * a[2].s2 - a[1].s2 * a[2].s1);
    d -= a[0].s1 * (a[1].s0 * a[2].s2 - a[1].s2 * a[2].s0);
    d += a[0].s2 * (a[1].s0 * a[2].s1 - a[1].s1 * a[2].s0);
    return d;
}

// Return diagonal vector of the supplied 3x3 matrix.
static fpreal3
diag3(const mat3 a)
{
    return (fpreal3)(a[0].s0, a[1].s1, a[2].s2);
}

// Set a to the 3x3 diagonal matrix defined by the entries of the vector diag.
static void
mat3diag(const fpreal3 diag, mat3 a)
{
    mat3zero(a);
    a[0].x = diag.x;
    a[1].y = diag.y;
    a[2].z = diag.z;
}

// Return the trace of the supplied 3x3 matrix.
static fpreal
trace3(const mat3 m)
{
    return vec3sum(diag3(m));
}

// Set 4x4 matrix A to the identity.
static void
mat4identity(mat4 *a)
{
    (*a).lo.lo = (fpreal4)(1.0f, 0.0f, 0.0f, 0.0f);
    (*a).lo.hi = (fpreal4)(0.0f, 1.0f, 0.0f, 0.0f);
    (*a).hi.lo = (fpreal4)(0.0f, 0.0f, 1.0f, 0.0f);
    (*a).hi.hi = (fpreal4)(0.0f, 0.0f, 0.0f, 1.0f);
}

// Multiply b * A where b is a 2-vector and A is a 4x4 matrix.
// This multiplication order matches VEX.
static fpreal2
mat4vec2mul(const mat4 a, const fpreal2 b)
{
    return b.x * a.lo.lo.xy +
           b.y * a.lo.hi.xy +
                 a.hi.hi.xy;
}

// Multiply b * A where b is a 3-vector and A is a 4x3 matrix,
// assuming a fourth component of the vector to be 0, i.e.
// the typical transformation of a 3d vector by a matrix.
// This multiplication order matches VEX.
static fpreal3
mat43vec3mul(const mat4 a, const fpreal3 b)
{
    fpreal4 result = b.x * a.lo.lo +
                    b.y * a.lo.hi +
                    b.z * a.hi.lo;
    return (fpreal3)(result.x, result.y, result.z);
}

// Multiply b * A where b is a 3-vector and A is a 4x4 matrix,
// assuming a fourth component of the vector to be 1, i.e.
// the typical transformation of a 3d point by a matrix.
// This multiplication order matches VEX.static fpreal3
static fpreal3
mat4vec3mul(const mat4 a, const fpreal3 b)
{
    fpreal4 result = b.x * a.lo.lo +
                    b.y * a.lo.hi +
                    b.z * a.hi.lo +
                    a.hi.hi;
    return (fpreal3)(result.x, result.y, result.z);
}

// Multiply b * A where b is a 4-vector and A is a 4x4 matrix.
// This multiplication order matches VEX.
static fpreal4
mat4vecmul(const mat4 a, const fpreal4 b)
{
    fpreal4 result = b.x * a.lo.lo +
                    b.y * a.lo.hi +
                    b.z * a.hi.lo +
                    b.w * a.hi.hi;
    return result;
}

// UT_Matrix4::solveColumn
// cp:          pivot_col and pivot_row
// c1, c2, c3:  other columns and rows
static void
mat4solvecol(fpreal (*matx)[4][4], int cp)
{
    fpreal      pivot_value_inverse;
    int         c1, c2, c3;

    switch (cp)
    {
        case 0: c1 = 1; c2 = 2; c3 = 3; break;
        case 1: c1 = 0; c2 = 2; c3 = 3; break;
        case 2: c1 = 0; c2 = 1; c3 = 3; break;
        case 3: c1 = 0; c2 = 1; c3 = 2; break;
    }

    // Here we will find the inverse of the pivot, set the pivot to 1,
    // and multiply the row which contains the pivot by the pivot
    // inverse. This might seem a little weird. The algorithm does it
    // this way so that it can gradually replace the input matrix
    // with its inverse.
    pivot_value_inverse = 1.0F/(*matx)[cp][cp];

    (*matx)[cp][cp] = pivot_value_inverse;
    (*matx)[cp][c1] *= pivot_value_inverse;
    (*matx)[cp][c2] *= pivot_value_inverse;
    (*matx)[cp][c3] *= pivot_value_inverse;

    // Now we subtract multiples of the pivot row from the other
    // rows in the matrix. This would be more familiar if the pivot
    // itself hadn't been set to 1 before the pivot row was multiplied
    // by the inverse (see above).

    (*matx)[c1][c1] -= (*matx)[cp][c1]*(*matx)[c1][cp];
    (*matx)[c1][c2] -= (*matx)[cp][c2]*(*matx)[c1][cp];
    (*matx)[c1][c3] -= (*matx)[cp][c3]*(*matx)[c1][cp];

    (*matx)[c2][c1] -= (*matx)[cp][c1]*(*matx)[c2][cp];
    (*matx)[c2][c2] -= (*matx)[cp][c2]*(*matx)[c2][cp];
    (*matx)[c2][c3] -= (*matx)[cp][c3]*(*matx)[c2][cp];

    (*matx)[c3][c1] -= (*matx)[cp][c1]*(*matx)[c3][cp];
    (*matx)[c3][c2] -= (*matx)[cp][c2]*(*matx)[c3][cp];
    (*matx)[c3][c3] -= (*matx)[cp][c3]*(*matx)[c3][cp];

    (*matx)[c1][cp] *= -pivot_value_inverse;
    (*matx)[c2][cp] *= -pivot_value_inverse;
    (*matx)[c3][cp] *= -pivot_value_inverse;
}

// UT_Matrix4::invert
// Linear equation solution by Gauss-Jordan elimination.
static int
mat4invert(fpreal16 *m)
{
    fpreal tol = 0.0;
    fpreal (*matx)[4][4] = (fpreal (*)[4][4]) m;
    
    int indexcol[4], indexrow[4];
    int pivot_row, pivot_col;

    // Check for the very common case of trivial column 3.
    const bool is_trivial_col3 =
        ((*matx)[0][3] == 0) &&
        ((*matx)[1][3] == 0) &&
        ((*matx)[2][3] == 0) &&
        ((*matx)[3][3] == 1);

    // We will need to keep track of which columns we have already found
    // pivots in. For this we use what is essentially a set of flags in the
    // array "pivoted_columns_flags". We initialize them to 0 here.
    bool pivoted_columns_flags[4] = {false, false, false, false};

    // In order to invert an n*n matrix, we will have to find n pivots, and
    // for each pivot reduce each row. We will keep track of which pivot and
    // set of reductions we are working on with "reduction"

    for (int reduction = 0; reduction < 4; reduction++)
    {
        fpreal pivot_value = 0;

        // This is the outer loop of the search for a pivot element
        // This loop finds the pivot element by choosing the element of the
        // matrix which has the largest absolute value of all elements of the
        // array not in columns/rows that contain a previously used pivot.
        for (int row = 0; row < 4; row++)
        {
            // if there hasn't already been a pivot in this row
            if ( !pivoted_columns_flags[row] )
            {
                // if there hasn't already been a pivot in column 0
                if ( !pivoted_columns_flags[0] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][0]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 0;
                    }
                }

                // if there hasn't already been a pivot in column 1
                if ( !pivoted_columns_flags[1] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][1]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 1;
                    }
                }

                // if there hasn't already been a pivot in column 2
                if ( !pivoted_columns_flags[2] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][2]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 2;
                    }
                }

                // if there hasn't already been a pivot in column 3
                if ( !pivoted_columns_flags[3] )
                {
                    const fpreal abs_element_value = fabs((*matx)[row][3]);

                    if ( abs_element_value > pivot_value )
                    {
                        pivot_value = abs_element_value;
                        pivot_row = row;
                        pivot_col = 3;
                    }
                }
            }
        }

        // Odd check here if the matrix is filled with nan's or infinities
        // Verify that the pivot we found is not 0
        if (pivot_value <= tol)
        {
            return 1;
        }

        // Now we have found the pivot element for the current reduction.
        // This element of the matrix is the largest element of all elements
        // not in rows and columns previously used for other pivots.

        // Record that we have found a pivot in the current column
        pivoted_columns_flags[pivot_col] = true;
        indexrow[reduction] = pivot_row;
        indexcol[reduction] = pivot_col;

        // The pivot must be on the diagonal before we can use it. If the
        // pivot we found wasn't already on the diagonal, we swap rows to
        // put it there now.
        if ( pivot_row != pivot_col )
        {
            swapf(&(*matx)[pivot_row][0], &(*matx)[pivot_col][0]);
            swapf(&(*matx)[pivot_row][1], &(*matx)[pivot_col][1]);
            swapf(&(*matx)[pivot_row][2], &(*matx)[pivot_col][2]);
            swapf(&(*matx)[pivot_row][3], &(*matx)[pivot_col][3]);
        }


        // Note that from here on, the pivot is on the diagonal, and thus
        // has the same row index as column index. In particular, while
        // we may have swapped the row the pivot is in, we have not changed
        // the column. Thus, the pivot location will be referred to using
        // pivot_col only.

        switch (pivot_col)
        {
            case 0: mat4solvecol(matx, 0); break;
            case 1: mat4solvecol(matx, 1); break;
            case 2: mat4solvecol(matx, 2); break;
            case 3: mat4solvecol(matx, 3); break;
        }
    }


    // Finally, we "unscramble" the matrix, which was scrambled by
    // row swapping. This will produce the actual inverse of the matrix.
    for (int reduction = 3; reduction >= 0; reduction--)
    {
        int irow = indexrow[reduction];
        int icol = indexcol[reduction];
        if ( irow != icol )
        {
            swapf(&(*matx)[0][irow], &(*matx)[0][icol]);
            swapf(&(*matx)[1][irow], &(*matx)[1][icol]);
            swapf(&(*matx)[2][irow], &(*matx)[2][icol]);
            swapf(&(*matx)[3][irow], &(*matx)[3][icol]);
        }
    }

    if (is_trivial_col3)
    {
        // Force column 3 to be exact if input column 3 was exactly trivial.
        (*matx)[0][3] = 0;
        (*matx)[1][3] = 0;
        (*matx)[2][3] = 0;
        (*matx)[3][3] = 1;
    }

    return 0;
}


static fpreal 
mat2det(const mat2 m)
{
    return m.x * m.w - m.y * m.z;
}

static fpreal 
mat2inv(const mat2 m, mat2 *minvout)
{
    fpreal det = mat2det(m);
    if (det == 0)
        return 0;

    *minvout = (mat2)(
        m.w, -m.y,
        -m.z, m.x
    ) / det;
    return det;
}

// 3x3 matrix inversion that returns zero if |det(m)|<=tol.
static fpreal 
mat3invtol(const mat3 m, mat3 minvout, fpreal tol)
{
    fpreal det = det3(m);
    if (fabs(det) <= tol)
        return 0;

    // Inverse by cofactor method.
    minvout[0] = (fpreal3)(
         mat2det((mat2)(m[1].yz, m[2].yz)),
        -mat2det((mat2)(m[0].yz, m[2].yz)),
         mat2det((mat2)(m[0].yz, m[1].yz))
    ) / det;
    minvout[1] = (fpreal3)(
        -mat2det((mat2)(m[1].xz, m[2].xz)),
         mat2det((mat2)(m[0].xz, m[2].xz)),
        -mat2det((mat2)(m[0].xz, m[1].xz))        
    ) / det;
    minvout[2] = (fpreal3)(
         mat2det((mat2)(m[1].xy, m[2].xy)),
        -mat2det((mat2)(m[0].xy, m[2].xy)),
         mat2det((mat2)(m[0].xy, m[1].xy))
    ) / det;

    return det;
}

// 3x3 matrix inversion that returns zero if |det(m)|==0.
static fpreal 
mat3inv(const mat3 m, mat3 minvout)
{
    return mat3invtol(m, minvout, 0);
}

static void
mat3scale(mat3 mout, const mat3 m, fpreal scale)
{
    mout[0] = m[0] * scale;
    mout[1] = m[1] * scale;
    mout[2] = m[2] * scale;
}

// In-place version of above.
static 
void mat3scaleip(mat3 A, fpreal scale)
{
    A[0] *= scale;
    A[1] *= scale;
    A[2] *= scale;
}

static void
mat3lincomb2(mat3 mout, const mat3 m1, fpreal scale1, const mat3 m2, fpreal scale2)
{
    mout[0] = m1[0] * scale1 + m2[0] * scale2;
    mout[1] = m1[1] * scale1 + m2[1] * scale2;
    mout[2] = m1[2] * scale1 + m2[2] * scale2;
}

// Rotates the incoming positions with the given angle in degrees.
// It's more efficient to make a rotation matrix if you're going to rotate multiple matrices by the same angle
static fpreal2
rotate2D(fpreal2 pos, fpreal angle)
{
    angle = -angle * M_PI_F/180.0f;
    fpreal ca;
    fpreal sa = sincos(angle, &ca);
    mat2 rot = (mat2)(ca, sa, -sa, ca);
    return mat2vecmul(rot, pos);
}

// Returns if matrix is Positive Definite using the Sylvester condition.
static int
mat3isPD(mat3 A)
{
    // Test determinants of each left submatrix.
    // 1x1 det
    if (A[0].x <= 0)
        return 0;
    // 2x2 det
    if (A[0].x * A[1].y - A[0].y * A[1].x <= 0)
        return 0;
    // 3x3 det        
    if (det3(A) <= 0)
        return 0;
    return 1;
}

// Force a matrix to be symmetric,
// usually to fix floating point error.
static void
mat3makesym(mat3 A)
{
    A[1].s0 = A[0].s1;
    A[2].s0 = A[0].s2;
    A[2].s1 = A[1].s2;
}

// Returns if the matrix is symmetric.
static int
mat3issym(mat3 A)
{
    return (A[1].s0 == A[0].s1 &&
            A[2].s0 == A[0].s2 && 
            A[2].s1 == A[1].s2);
}

// Solve a 3x3 Symmetric Positive Definite matrix
// using the LDL^T decomposition.
static fpreal3
mat3solve_LDLT(mat3 a, fpreal3 b)
{
    // Compute LDL^T decomposition
    fpreal D1 = a[0].x;
    fpreal L21 = a[1].x / D1;
    fpreal L31 = a[2].x / D1;
    fpreal D2 = a[1].y - L21 * L21 * D1;
    fpreal L32 = (a[2].y - L21 * L31 * D1) / D2;
    fpreal D3 = a[2].z - (L31 * L31 * D1 + L32 * L32 * D2);

    // Forward substitution: Solve Ly = b
    fpreal y1 = b.x;
    fpreal y2 = b.y - L21 * y1;
    fpreal y3 = b.z - L31 * y1 - L32 * y2;

    // Diagonal solve: Solve Dz = y
    fpreal z1 = y1 / D1;
    fpreal z2 = y2 / D2;
    fpreal z3 = y3 / D3;

    // Backward substitution: Solve L^T x = z
    fpreal3 x;
    x.z = z3;
    x.y = z2 - L32 * x.z;
    x.x = z1 - L21 * x.y - L31 * x.z;

    return x;
}

#ifndef NO_DOUBLE_SUPPORT
// Double versions of subset of the above functions
// (mainly for shapematching ATM).

typedef double3 mat3d[3];

static void
mat3fromcolsd(const double3 c0, const double3 c1, const double3 c2, mat3d m)
{
    m[0] = (double3)(c0.s0, c1.s0, c2.s0);
    m[1] = (double3)(c0.s1, c1.s1, c2.s1);
    m[2] = (double3)(c0.s2, c1.s2, c2.s2);
}

static void
transpose3d(const mat3d a, mat3d b)
{
    mat3fromcolsd(a[0], a[1], a[2], b);
}

#endif

#endif
#ifndef __RANDOM_H
#define __RANDOM_H

/******************************************************************************
 * HDK-consistent floor, integer hash, and fast random number generation.
 ******************************************************************************/

/// Returns the largest representable integer no greater than the given input.
static float
SYSfloorIL(float val)
{
    uint tmp = as_uint(val);
    uint shift = (tmp >> 23) & 0xff;

    if(shift < 0x7f)
    {
        return (tmp > 0x80000000) ? -1.0F : 0.0F;
    }
    else if(shift < 0x96)
    {
        uint mask = 0xffffffff << (0x96 - shift);
        if(tmp & 0x80000000)
        {
            if((tmp & ~mask) & 0x7fffff)
            {
                tmp &= mask;
                return as_float(tmp) - 1;
            }
            else
            {
                return val;
            }
        }
        else
        {
            return as_float(tmp & mask);
        }
    }
    else
    {
        return val;
    }
}

/// Consistent integer hash with the HDK.
static uint
SYSwang_inthash(uint key)
{
    key += ~(key << 16);
    key ^=  (key >>  5);
    key +=  (key <<  3);
    key ^=  (key >> 13);
    key += ~(key <<  9);
    key ^=  (key >> 17);
    return key;
}

/// Generates a uniform random number in the 0-1 range from the given seed and
/// updates the seed.
static float
SYSfastRandom(uint* seed)
{
    uint temp;
    *seed = (*seed) * 1664525 + 1013904223;
    temp = 0x3f800000 | (0x007fffff & (*seed));
    return as_float(temp) - 1.0f;
}

/******************************************************************************
 * Hashing macros for varying number of inputs.
 ******************************************************************************/

/// Conversion from a single float to an integer, for the clamped and unclamped
/// cases.
#define C_HASH1(x) ((int) SYSfloorIL(x))
#define U_HASH1(x) (as_uint(x))

/// Hash functions for 2-4 integers, used by clamped float hashing.
#define HASH2I(x, y) ((x^0xffffdead) * (y^0xffffc0de))
#define HASH3I(x, y, z) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169))
#define HASH4I(x, y, z, w) ((x^0xffff3ce3) * (y^0xffff7ba5) * (z^0xffffd169) \
                                           * (w^0xffff0397))
/// These macros declare hashing functions for clamped floating point numbers.
#define C_HASH2(x, y) HASH2I(C_HASH1(x), C_HASH1(y))
#define C_HASH3(x, y, z) HASH3I(C_HASH1(x), C_HASH1(y), C_HASH1(z))
#define C_HASH4(x, y, z, w) HASH4I(C_HASH1(x), C_HASH1(y), C_HASH1(z), \
                                   C_HASH1(w))

/// This macro hashes 2 integers, used by generic float hashing.
#define U_HASH2_RAW(x, y) y + SYSwang_inthash(x)
/// These macros declare hashing functions for generic floating point numbers.
#define U_HASH2(x, y) U_HASH2_RAW(U_HASH1(x), U_HASH1(y))
#define U_HASH3(x, y, z) U_HASH2_RAW(U_HASH2(x, y), U_HASH1(z))
#define U_HASH4(x, y, z, w) U_HASH2_RAW(U_HASH3(x, y, z), U_HASH1(w))

/******************************************************************************
 * Helper macros to hash inputs and generate outputs.
 ******************************************************************************/

/// Macros to hash the given number of floating point variables and store the
/// result in a new integer called hash.
#define C_1_HASH uint hash = SYSwang_inthash(C_HASH1(x));
#define C_2_HASH uint hash = SYSwang_inthash(C_HASH2(x, y));
#define C_3_HASH uint hash = SYSwang_inthash(C_HASH3(x, y, z));
#define C_4_HASH uint hash = SYSwang_inthash(C_HASH4(x, y, z, w));
#define U_1_HASH int hash = SYSwang_inthash(U_HASH1(x));
#define U_2_HASH int hash = SYSwang_inthash(U_HASH2(x, y));
#define U_3_HASH int hash = SYSwang_inthash(U_HASH3(x, y, z));
#define U_4_HASH int hash = SYSwang_inthash(U_HASH4(x, y, z, w));
/// Macros to return a vector of K random numbers; hash integer variable must be
/// declared and initialized.
#define RET_1_RAND return SYSfastRandom((uint*) &hash);
#define RET_2_RAND return (float2)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_3_RAND return (float3)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));
#define RET_4_RAND return (float4)(SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash), \
                                   SYSfastRandom((uint*) &hash));

/******************************************************************************
 * Generation of final VEX_equivalent random_fhash() functions.
 ******************************************************************************/

static int VEXrandom_fhash_1(float x)
{
    U_1_HASH
    return hash;
}
static int VEXrandom_fhash_2(float x, float y)
{
    U_2_HASH
    return hash;
}
static int VEXrandom_fhash_3(float x, float y, float z)
{
    U_3_HASH
    return hash;
}
static int VEXrandom_fhash_4(float x, float y, float z, float w)
{
    U_4_HASH
    return hash;
}

/******************************************************************************
 * Generation of the final VEX-equivalent random() functions with float inputs.
 ******************************************************************************/

/// Macro that generates the code for random functions. NUM should be 2-4
/// (number of random numbers to return).
#define CREATE_RANDOM(NUM) \
static float ## NUM VEXrandom_1_ ## NUM(float x) \
{ \
    C_1_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_2_ ## NUM(float x, float y) \
{ \
    C_2_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_3_ ## NUM(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrandom_4_ ## NUM(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point random.
#define CREATE_RANDOM_FLOAT \
static float VEXrandom_1_1(float x) \
{ \
    C_1_HASH \
    RET_1_RAND \
} \
static float VEXrandom_2_1(float x, float y) \
{ \
    C_2_HASH \
    RET_1_RAND \
} \
static float VEXrandom_3_1(float x, float y, float z) \
{ \
    C_3_HASH \
    RET_1_RAND \
} \
static float VEXrandom_4_1(float x, float y, float z, float w) \
{ \
    C_4_HASH \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RANDOM_FLOAT
CREATE_RANDOM(2)
CREATE_RANDOM(3)
CREATE_RANDOM(4)

/******************************************************************************
 * Generation of the final VEX-equivalent rand() functions.
 ******************************************************************************/

/// Macro that generates the code for rand functions. NUM should be 2-4 (number
/// of random numbers to return).
#define CREATE_RAND(NUM) \
static float ## NUM VEXrand_1_ ## NUM(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_2_ ## NUM(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_3_ ## NUM(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
} \
static float ## NUM VEXrand_4_ ## NUM(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_ ## NUM ## _RAND \
}
/// Macro that generates the code for returning a single floating point rand.
#define CREATE_RAND_FLOAT \
static float VEXrand_1_1(float x) \
{ \
    U_1_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_2_1(float x, float y) \
{ \
    U_2_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_3_1(float x, float y, float z) \
{ \
    U_3_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
} \
static float VEXrand_4_1(float x, float y, float z, float w) \
{ \
    U_4_HASH \
    hash = (int) SYSwang_inthash(hash); \
    RET_1_RAND \
}

/// Create the functions.
CREATE_RAND_FLOAT
CREATE_RAND(2)
CREATE_RAND(3)
CREATE_RAND(4)

#endif

#ifndef __IMX_H__
#define __IMX_H__

/******************************************************************************
 * VERBOSITY OPTIONS
 ******************************************************************************/
 
// By defining certain symbols before including this header, different error
// reporting options can be enabled. These options are listed in the table
// below.
//      SYMBOL                  ||          EFFECT
//  CHECK_RANGE                 ||  Report out-of-range indexing of buffers
//  CHECK_STORAGE_TYPE_READ     ||  Reports attempts to read floating point
//                              ||  values from integer layers or integer values
//                              ||  from floating point layers.
//  CHECK_STORAGE_TYPE_WRITE    ||  Reports attempts to write floating point
//                              ||  values to integer layers or integer values
//                              ||  to floating point layers.
//  CHECK_STORAGE_TYPE_CON      ||  Report if IMX_Layer storage type is not equal
//                              ||  to the (compile-time constant) storage argument
//  CHECK_CHANNEL_COUNT_CON     ||  Report if IMX_Layer channels is not equal
//                              ||  to the (compile-time constant) channels argument
// Note that the checks are done only when the respective symbol is defined, so
// these should only be used for validation and debugging purposes.

/******************************************************************************
 * STRUCTURES
 ******************************************************************************/

// Scale-translate transform.
typedef float4 STXform;

// Border type for a layer.
typedef enum
{
    IMX_CONSTANT,
    IMX_CLAMP,
    IMX_MIRROR,
    IMX_WRAP
} BorderType;

typedef enum
{
    IMX_TYPEINFO_NONE,
    IMX_TYPEINFO_COLOR,
    IMX_TYPEINFO_POSITION,
    IMX_TYPEINFO_VECTOR,
    IMX_TYPEINFO_NORMAL,
    IMX_TYPEINFO_OFFSETNORMAL,
    IMX_TYPEINFO_TEXTURE_COORD,
    IMX_TYPEINFO_ID,
    IMX_TYPEINFO_MASK,
    IMX_TYPEINFO_SDF,
    IMX_TYPEINFO_HEIGHT
} TypeInfoType;

// Type of data stored in a layer.
typedef enum
{
    INT8,
    INT16,
    INT32,
    FLOAT16,
    FLOAT32,
    FIXED8,
    FIXED16,
    // dummy values used to indicate IMX_Buffer::isConstant():
    CONSTANT_INT8,
    CONSTANT_INT16,
    CONSTANT_INT32,
    CONSTANT_FLOAT16,
    CONSTANT_FLOAT32,
    CONSTANT_FIXED8,
    CONSTANT_FIXED16
} StorageType;

// Type of projection.
typedef enum
{
    IMX_PROJ_ORTHOGRAPHIC,
    IMX_PROJ_PERSPECTIVE,
} ProjectionType;


// A structure containing metadata for a layer.
typedef struct
{
    float16                     image_to_world;
    float16                     world_to_image;
    float16                     camera_to_world;

    STXform                     buffer_to_image;
    STXform                     image_to_buffer;
    STXform                     buffer_to_pixel;
    float3                      camera_image_pos;

    float4                      default_f;
    int4                        default_i;

    int2                        resolution;
    int                         channels;
    int                         stride_x, stride_y;

    BorderType                  border;
    // Typeinfo is a hint and should only affect computation in the rarest
    // of situations.
    TypeInfoType                typeinfo;
    StorageType                 storage;
    ProjectionType              projection;
} IMX_Stat;

// A structure encapsulating a layer.
typedef struct
{
    global void* restrict       data;
    global IMX_Stat* restrict   stat;
} IMX_Layer;

/// multiply v by scale+translate xform
static float2 applySTXform(STXform, float2 v);
static float2 applySTXformInverse(STXform, float2 v);
static float2 applySTXformVec(STXform, float2 v);
static float2 applySTXformInverseVec(STXform, float2 v);

/// Space transfomrations;
static float2 bufferToImage(global const IMX_Stat* restrict, float2 xy);
static float2 imageToBuffer(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToPixel(global const IMX_Stat* restrict, float2 xy);
static float2 pixelToBuffer(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToTexture(global const IMX_Stat* restrict, float2 xy);
static float2 textureToBuffer(global const IMX_Stat* restrict, float2 xy);

static float3 imageToWorld(global const IMX_Stat* restrict, float2 xy);
static float3 image3ToWorld(global const IMX_Stat* restrict, float3 xy);
static float2 worldToImage(global const IMX_Stat* restrict, float3 xyz);
static float3 worldToImage3(global const IMX_Stat* restrict, float3 xyz);

/// Vector variants.
static float2 bufferToImageVec(global const IMX_Stat* restrict, float2 xy);
static float2 imageToBufferVec(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToPixelVec(global const IMX_Stat* restrict, float2 xy);
static float2 pixelToBufferVec(global const IMX_Stat* restrict, float2 xy);
static float2 bufferToTextureVec(global const IMX_Stat* restrict, float2 xy);
static float2 textureToBufferVec(global const IMX_Stat* restrict, float2 xy);

static float3 imageToWorldVec(global const IMX_Stat* restrict, float2 xy);
static float3 image3ToWorldVec(global const IMX_Stat* restrict, float3 xy);
static float2 worldToImageVec(global const IMX_Stat* restrict, float3 xyz);
static float3 worldToImage3Vec(global const IMX_Stat* restrict, float3 xyz);

// The remaining functions are for implementing @ substitutions:
//
//  int @ix, @iy, @ixy                  // output buffer coordinate
//  int @xres                           // output buffer width
//  int @yres                           // output buffer height
//  int2 @res                           // output buffer (width,height)
//  int2 @tilesize                      // tile dimensions passed to CE_Snippet::execute()
// 
// Cur location: .image .pixel .texture suffixes specify space, image default
// @P supports .world as well.
//  float2 @P                           // image coordinate of output pixel
//  float2 @dPdx                        // derivative of @P per @ix
//  float2 @dPdy                        // derivative of @P per @iy
//  float2 @dPdxy                       // (@dPdx.x,@dPdy.y) rectangle for area
//                                      // sampling
//
// 'name' is replaced with the name of a layer binding:
//  void* @name.data                    // raw buffer data
//  IMX_Stat* @name.stat
//  int @name.xres                      // @name's buffer width
//  int @name.yres                      // @name's buffer height
//  float2 @name.res                    // @name's (width,height)
//  BorderType @name.border             // border type, compile-time constant
//  StorageType @name.storage           // data type, compile-time constant
//  int @name.channels                  // # of channels, compile-time constant
//  int @name.tuplesize                 // # of channels, compile-time constant
//
// Space transforms:
//  float2 @name.imageToBuffer(float2) 
//  float2 @name.bufferToImage(float2) 
//  float2 @name.pixelToBuffer(float2) 
//  float2 @name.bufferToPixel(float2) 
//  float2 @name.textureToBuffer(float2) 
//  float2 @name.bufferToTexture(float2) 
//  float3 @name.imageToWorld(float2)
//  float3 @name.image3ToWorld(float3)
//  float2 @name.worldToImage(float3)
//  float3 @name.worldToImage3(float3)
//
//  bool @name.bound                    // same as #ifdef HAS_name
//
//  T @name.bufferIndex(int2)           // value of buffer pixel, does tiling/borders
//  T @name.bufferSample(float2)        // bilinear interpolated (nearest for int)
//  T @name.imageNearest(float2)        // bufferIndex(rint(imageToBuffer(xy))
//  T @name.imageSample(float2)         // bufferSample(imageToBuffer(xy))
//  T @name.textureNearest(float2)      // bufferIndex(rint(textureToBuffer(xy))
//  T @name.textureSample(float2)       // bufferSample(textureToBuffer(xy))
//  T @name.worldSample(float3)         // bufferSample(imageToBuffer(worldToImage(xyz)))
//  T @name.worldNearest(float3)         // bufferIndex(rint(imageToBuffer(worldToImage(xyz))))
//  T @name                             // @name.imageSample(@P)
//  void @name.set(T v)                 // same as @name.setIndex((int2)(@ix,@iy), v)
//  void @name.setIndex(int2, T v)      // store value of buffer pixel, no test for out of range!
//
// Where T is not int:
//  T @name.dCdx(float2)                // derivative of @name.imageSample() per @ix
//  T @name.dCdx                        // @name.dCdx(@P)
//  T @name.dCdy(float2)                // derivative of @name.imageSample() per @iy
//  T @name.dCdy                        // @name.dCdy(@P)

// #include "imx_internal.h"
#endif

#ifndef __IMX_INTERNAL_H__
#define __IMX_INTERNAL_H__

// #include <matrix.h>

typedef float                   float1;
typedef int                     int1;

/// Converts image coordinates to linear index.
static int
_linearIndex(global const IMX_Stat* restrict stat, int2 xy)
{
#ifdef CHECK_RANGE
    if (xy.x < 0 || xy.x >= stat->resolution.x ||
        xy.y < 0 || xy.y >= stat->resolution.y)
        printf("Error: converting an invalid 2D index at %v2d to linear;"
               "resolution was %v2d.\n", xy, stat->resolution);
#endif
    return xy.x * stat->stride_x + xy.y * stat->stride_y;
}

/// Splits the given coordinates into integer and fractional parts.
static void
_splitCoordinates(float2 xy, int2* ix_p, float2* fx_p)
{
    float2 f;
    *fx_p = fract(xy, &f);
    *ix_p = convert_int2_sat(f);
}

/// Wraps the given coordinates for the specified resolution.
static int2
_wrapCoordinates(int2 xy, int2 res)
{
    int2 p = xy % res;
    return select(p, p + res, p < 0);
}

static int2
_mirrorCoordinates(int2 xy, int2 res)
{
    int2 res2 = res * 2;
    int2 p = _wrapCoordinates(xy, res2);
    return select(p, res2 - p - 1, p >= res);
}

/// Mirrors the given coordinates for the specified resolution. mirrored0 and
/// mirrored1 will have the mirrored versions of x and x+1, respectively.
static void
_mirrorCoordinates2(int2 xy, int2 res, int2* mirrored0, int2* mirrored1)
{
    int2 res2 = res * 2;
    int2 p = _wrapCoordinates(xy, res2);
    *mirrored0 = select(p, res2 - p - 1, p >= res);
    p = _wrapCoordinates(xy + 1, res2);
    *mirrored1 = select(p, res2 - p - 1, p >= res);
}

static bool
_outside(int2 xy, int2 res)
{
    return any(xy < 0) || any(xy >= res);
}

#ifdef CHECK_STORAGE_TYPE_CON
__constant char *
_getStorageName(StorageType storage)
{
    switch (storage)
    {
    case INT8:
        return "INT8";
    case INT16:
        return "INT16";
    case INT32:
        return "INT32";
    case FLOAT16:
        return "FLOAT16";
    case FLOAT32:
        return "FLOAT32";
    case FIXED8:
        return "FIXED8";
    case FIXED16:
        return "FIXED16";
    }
}
#define _CHECK_STORAGE(what) \
    if (storage != layer->stat->storage) \
        printf("Error: %s(%s), layer is %s\n", what, \
               _getStorageName(storage), _getStorageName(layer->stat->storage));
#else
#define _CHECK_STORAGE(what)
#endif

#ifdef CHECK_CHANNEL_COUNT_CON
#define _CHECK_CHANNEL(what) \
    if (channels != layer->stat->channels) \
        printf("Error: %s(channels=%d), layer is %d channels\n", \
               what, channels, layer->stat->channels);
#else
#define _CHECK_CHANNEL(what)
#endif

#ifdef CHECK_RANGE
#define _CHECK_RANGE(what) \
    if (index < 0 || index >= stat->resolution.y * stat->stride_y) \
        printf("Error: %s index %d out of range\n", what, index);
#else
#define _CHECK_RANGE(what)
#endif

#define CHECK_STAT(what) _CHECK_STORAGE(what) _CHECK_CHANNEL(what) _CHECK_RANGE(what)

#ifdef CHECK_STORAGE_TYPE_WRITE
#define WRITE_ERROR_I() printf("Error: writing integer values to a floating point layer.\n"); return
#define WRITE_ERROR_F() printf("Error: writing floating point values to an integer layer.\n"); return
#else
#define WRITE_ERROR_I() return
#define WRITE_ERROR_F() return
#endif

#ifdef CHECK_STORAGE_TYPE_READ
#define READ_ERROR_I() printf("Error: reading integer values from a floating point layer.\n"); return layer->stat->default_i
#define READ_ERROR_F() printf("Error: reading floating point values from an integer layer.\n"); return layer->stat->default_f
#else
#define READ_ERROR_I() return layer->stat->default_i
#define READ_ERROR_F() return layer->stat->default_f
#endif

// TODO: make sure rounding mode is good. (FIXED_POINT_STORAGE)
/// 8-bit fixed point conversion macros.
#define TO_FIXED8_SCALE 255
#define FROM_FIXED8_SCALE 0.00392156862f
#define FP_TO_FIXED8(v) convert_uchar_rte(clamp(v, 0.0f, 1.0f) \
                                          * TO_FIXED8_SCALE)
#define FP_TO_FIXED8_v(v, COMP) convert_uchar ## COMP ## _rte( \
    clamp(v, 0.0f, 1.0f) * TO_FIXED8_SCALE)
#define FIXED8_TO_FP(v) convert_float(v) * FROM_FIXED8_SCALE
#define FIXED8_TO_FP_v(v, COMP) convert_float ## COMP (v) \
    * FROM_FIXED8_SCALE
/// 16-bit fixed point conversion macros.
#define TO_FIXED16_SCALE 32767
#define FROM_FIXED16_SCALE 0.0000305185f
#define FP_TO_FIXED16(v) convert_short_rte(clamp(v, -1.0f, 1.0f) \
                                           * TO_FIXED16_SCALE)
#define FP_TO_FIXED16_v(v, COMP) convert_short ## COMP ## _rte( \
    clamp(v, -1.0f, 1.0f) * TO_FIXED16_SCALE)
#define FIXED16_TO_FP(v) convert_float(v) * FROM_FIXED16_SCALE
#define FIXED16_TO_FP_v(v, COMP) convert_float ## COMP (v) \
    * FROM_FIXED16_SCALE

static void
_setIndexLinI1a(IMX_Layer* layer, int index, int v, StorageType storage)
{
    switch (storage)
    {
    case INT8:
        ((global char*) layer->data)[index] = (char) v;
        return;
    case INT16:
        ((global short*) layer->data)[index] = (short) v;
        return;
    case INT32:
        ((global int*) layer->data)[index] = v;
        return;
    case FLOAT16:
        vstore_half_rte((float) v, index, (global half*)layer->data);
        break;
    case FLOAT32:
        ((global float*) layer->data)[index] = v;
        break;
    // TODO: should these clamp to 0 and 1 and convert to fixed point value?
    // (FIXED_POINT_STORAGE)
    case FIXED8:
        ((global uchar*) layer->data)[index] = (uchar) v;
        return;
    case FIXED16:
        ((global short*) layer->data)[index] = (short) v;
        return;
    default:
        WRITE_ERROR_I();
    }
}

static void
_setIndexLinF1a(IMX_Layer* layer, int index, float v, StorageType storage)
{
    switch (storage)
    {
    case FLOAT16:
        vstore_half_rte(v, index, (global half*) layer->data);
        break;
    case FLOAT32:
        ((global float*) layer->data)[index] = v;
        break;
    case FIXED8:
        ((global uchar*) layer->data)[index] = FP_TO_FIXED8(v);
        break;
    case FIXED16:
        ((global short*) layer->data)[index] = FP_TO_FIXED16(v);
        break;
    default:
        _setIndexLinI1a(layer, index, convert_int_sat_rtn(v+0.5f), storage);
    }
}

static void
_setIndexLinF2a(IMX_Layer* layer, int index, float2 v, StorageType storage)
{
    switch (storage)
    {
    case FLOAT16:
        vstore_half2_rte(v, index, (global half*) layer->data);
        break;
    case FLOAT32:
        vstore2(v, index, (global float*) layer->data);
        break;
    case FIXED8:
        vstore2(FP_TO_FIXED8_v(v, 2), index, (global uchar*) layer->data);
        break;
    case FIXED16:
        vstore2(FP_TO_FIXED16_v(v, 2), index, (global short*) layer->data);
        break;
    default:
        WRITE_ERROR_F();
    }
}

static void
_setIndexLinF3a(IMX_Layer* layer, int index, float3 v, StorageType storage)
{
    switch (storage)
    {
    case FLOAT16:
        vstore_half3_rte(v.xyz, index, (global half*) layer->data);
        break;
    case FLOAT32:
        vstore3(v.xyz, index, (global float*) layer->data);
        break;
    case FIXED8:
        vstore3(FP_TO_FIXED8_v(v, 3), index, (global uchar*) layer->data);
        break;
    case FIXED16:
        vstore3(FP_TO_FIXED16_v(v, 3), index, (global short*) layer->data);
        break;
    default:
        WRITE_ERROR_F();
    }
}

static void
_setIndexLinF4(IMX_Layer* layer, int index, float4 v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF4");
    switch (channels)
    {
    case 1:
        _setIndexLinF1a(layer, index, v.x, storage);
        break;
    case 2:
        _setIndexLinF2a(layer, index, v.xy, storage);
        break;
    case 3:
        _setIndexLinF3a(layer, index, v.xyz, storage);
        break;
    default:
        switch (storage)
        {
        case FLOAT16:
            vstore_half4_rte(v, index, (global half*) layer->data);
            break;
        case FLOAT32:
            vstore4(v, index, (global float*) layer->data);
            break;
        case FIXED8:
            vstore4(FP_TO_FIXED8_v(v, 4), index, (global uchar*) layer->data);
            break;
        case FIXED16:
            vstore4(FP_TO_FIXED16_v(v, 4), index, (global short*) layer->data);
            break;
        default:
            WRITE_ERROR_F();
        }
        break;
    }
}

static void
_setIndexLinI1(IMX_Layer* layer, int index, int v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexI1");
    if (channels == 1)
        _setIndexLinI1a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, v, storage, channels);
}

static void
_setIndexLinF1(IMX_Layer* layer, int index, float v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF1");
    if (channels == 1)
        _setIndexLinF1a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, v, storage, channels);
}

static void
_setIndexLinF2(IMX_Layer* layer, int index, float2 v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF2");
    if (channels == 2)
        _setIndexLinF2a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, (float4)(v,0.0f,1.0f), storage, channels);
}

static void
_setIndexLinF3(IMX_Layer* layer, int index, float3 v, StorageType storage,
               int channels)
{
    CHECK_STAT("setIndexF3");
    if (channels == 3)
        _setIndexLinF3a(layer, index, v, storage);
    else
        _setIndexLinF4(layer, index, (float4)(v,1.0f), storage, channels);
}

////////////////////////////////////////////////////////////////////////////////

static int
_bufferIndexLinI1(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexI1");

    switch (storage)
    {
    case INT8:
        return ((global char*) layer->data)[index * channels];
    case INT16:
        return ((global short*) layer->data)[index * channels];
    case INT32:
        return ((global int*) layer->data)[index * channels];
    case CONSTANT_INT8:
        return *((global char*) layer->data);
    case CONSTANT_INT16:
        return *((global short*) layer->data);
    case CONSTANT_INT32:
        return *((global int*) layer->data);
    default:
        READ_ERROR_I().x;
    }
}

static float
_bufferIndexLinF1(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF1");

    switch (storage)
    {
    case FLOAT16:
        return vload_half(index*channels, (global half*) layer->data);
    case FLOAT32:
        return ((global float*) layer->data)[index*channels];
    case FIXED8:
        return FIXED8_TO_FP(((global uchar*) layer->data)[index*channels]);
    case FIXED16:
        return FIXED16_TO_FP(((global short*) layer->data)[index*channels]);
    case CONSTANT_FLOAT16:
        return vload_half(0, (global half*) layer->data);
    case CONSTANT_FLOAT32:
        return ((global float*) layer->data)[0];
    case CONSTANT_FIXED8:
        return FIXED8_TO_FP(((global uchar*) layer->data)[0]);
    case CONSTANT_FIXED16:
        return FIXED16_TO_FP(((global short*) layer->data)[0]);
    default:
        return _bufferIndexLinI1(layer, index, storage, channels);
    };
}

static float2
_bufferIndexLinF2(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF2");

    switch (channels)
    {
    case 1:
        return _bufferIndexLinF1(layer, index, storage, channels);
    default:
        switch (storage)
        {
        case FLOAT16:
            return vload_half2(0, (global half*) layer->data + index * channels);
        case FLOAT32:
            return vload2(0, (global float*) layer->data + index * channels);
        case FIXED8:
            return FIXED8_TO_FP_v(
                    vload2(0, (global uchar*) layer->data + index * channels),
                    2);
        case FIXED16:
            return FIXED16_TO_FP_v(
                    vload2(0, (global short*) layer->data + index * channels),
                    2);
        case CONSTANT_FLOAT16:
            return vload_half2(0, (global half*) layer->data);
        case CONSTANT_FLOAT32:
            return vload2(0, (global float*) layer->data);
        case CONSTANT_FIXED8:
            return FIXED8_TO_FP_v(vload2(0, (global uchar*) layer->data), 2);
        case CONSTANT_FIXED16:
            return FIXED16_TO_FP_v(vload2(0, (global short*) layer->data), 2);
        default:
            READ_ERROR_F().xy;
        }
    }
}

static float3
_bufferIndexLinF3(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF3");

    switch (channels)
    {
    case 1:
        return _bufferIndexLinF1(layer, index, storage, channels);
    case 2:
        return (float3)(_bufferIndexLinF2(layer, index, storage, channels), 0.0f);
    default:
        switch (storage)
        {
        case FLOAT16:
            return vload_half3(0, (global half*) layer->data + index * channels);
        case FLOAT32:
            return vload3(0, (global float*) layer->data + index * channels);
        case FIXED8:
            return FIXED8_TO_FP_v(
                    vload3(0, (global uchar*) layer->data + index * channels),
                    3);
        case FIXED16:
            return FIXED16_TO_FP_v(
                    vload3(0, (global short*) layer->data + index * channels),
                    3);
        case CONSTANT_FLOAT16:
            return vload_half3(0, (global half*) layer->data);
        case CONSTANT_FLOAT32:
            return vload3(0, (global float*) layer->data);
        case CONSTANT_FIXED8:
            return FIXED8_TO_FP_v(vload3(0, (global uchar*) layer->data), 3);
        case CONSTANT_FIXED16:
            return FIXED16_TO_FP_v(vload3(0, (global short*) layer->data), 3);
        default:
            READ_ERROR_F().xyz;
        }
    }
}

static float4
_bufferIndexLinF4(const IMX_Layer* layer, int index, StorageType storage,
                  int channels)
{
    CHECK_STAT("bufferIndexF4");

    switch (channels)
    {
    case 1:
        return _bufferIndexLinF1(layer, index, storage, channels);
    case 2:
        return (float4)(_bufferIndexLinF2(layer, index, storage, channels), 0.0f, 1.0f);
    case 3:
        return (float4)(_bufferIndexLinF3(layer, index, storage, channels), 1.0f);
    default:
        switch (storage)
        {
        case FLOAT16:
            return vload_half4(index, (global half*) layer->data);
        case FLOAT32:
            return vload4(index, (global float*) layer->data);
        case FIXED8:
            return FIXED8_TO_FP_v(vload4(index, (global uchar*) layer->data),
                                  4);
        case FIXED16:
            return FIXED16_TO_FP_v(vload4(index, (global short*) layer->data),
                                   4);
        case CONSTANT_FLOAT16:
            return vload_half4(0, (global half*) layer->data);
        case CONSTANT_FLOAT32:
            return vload4(0, (global float*) layer->data);
        case CONSTANT_FIXED8:
            return FIXED8_TO_FP_v(vload4(0, (global uchar*) layer->data), 4);
        case CONSTANT_FIXED16:
            return FIXED16_TO_FP_v(vload4(0, (global short*) layer->data), 4);
        default:
            READ_ERROR_F();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#define _IMPLEMENT_I(DIM)                                            \
                                                                     \
static int ## DIM                                                    \
bufferIndexI ## DIM (const IMX_Layer* layer, int2 xy,                \
                     BorderType border, StorageType storage,         \
                     int channels)                                   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
    {                                                                \
        switch (border)                                              \
        {                                                            \
        case (IMX_CONSTANT):                                         \
            return 0.0f;                                             \
        case (IMX_CLAMP):                                            \
            xy = clamp(xy, (int2)(0), res - 1);                      \
            break;                                                   \
        case (IMX_MIRROR):                                           \
            xy = _mirrorCoordinates(xy, res);                        \
            break;                                                   \
        case (IMX_WRAP): default:                                    \
            xy = _wrapCoordinates(xy, res);                          \
            break;                                                   \
        }                                                            \
    }                                                                \
    return _bufferIndexLinI ## DIM (                                 \
        layer, _linearIndex(layer->stat, xy), storage, channels);    \
}                                                                    \
                                                                     \
static int ## DIM                                                    \
bufferSampleI ## DIM (const IMX_Layer* layer, float2 xy,             \
                      BorderType border, StorageType storage,        \
                      int channels)                                  \
{                                                                    \
    int2 c = convert_int2_sat_rtn(xy+0.5f);                          \
    return bufferIndexI ## DIM (layer, c, border, storage, channels);\
}                                                                    \
                                                                     \
static void                                                          \
_setIndexI ## DIM(IMX_Layer *layer, int2 xy, int ## DIM val, StorageType storage, int channels)   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
        return;                                                      \
    _setIndexLinI ## DIM(layer, _linearIndex(layer->stat, xy), val, storage, channels); \
}                                                                    \
/**/


_IMPLEMENT_I(1)
#undef _IMPLEMENT_I

#define _IMPLEMENT_F(DIM)                                            \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _CN(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 res = stat->resolution;                                     \
    float ## DIM k = 0.0f;                                           \
    float ## DIM v00 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    ix.x++;                                                          \
    float ## DIM v10 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    ix.y++;                                                          \
    float ## DIM v11 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    ix.x--;                                                          \
    float ## DIM v01 = _outside(ix, res) ? k : _bufferIndexLinF ## DIM ( \
        layer, _linearIndex(stat, ix), storage, channels);           \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _CL(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 res1 = stat->resolution - 1;                                \
    int2 ix0 = clamp(ix, (int2)(0), res1);                           \
    int2 ix1 = clamp(ix+1, (int2)(0), res1);                         \
    float ## DIM v00 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix0), storage, channels);          \
    float ## DIM v10 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix1.x, ix0.y)), storage, channels); \
    float ## DIM v01 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix0.x, ix1.y)), storage, channels); \
    float ## DIM v11 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix1), storage, channels);          \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _MR(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 ix0, ix1;                                                   \
    _mirrorCoordinates2(ix, stat->resolution, &ix0, &ix1);           \
    float ## DIM v00 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix0), storage, channels);          \
    float ## DIM v10 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix1.x, ix0.y)), storage, channels); \
    float ## DIM v01 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix0.x, ix1.y)), storage, channels); \
    float ## DIM v11 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix1), storage, channels);          \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
_bufferSampleF ## DIM ## _WR(const IMX_Layer* layer, float2 xy,      \
                             StorageType storage, int channels)      \
{                                                                    \
    int2 ix;                                                         \
    float2 fx;                                                       \
    _splitCoordinates(xy, &ix, &fx);                                 \
    global const IMX_Stat* restrict stat = layer->stat;              \
    int2 ix0 = _wrapCoordinates(ix, stat->resolution);               \
    int2 ix1 = _wrapCoordinates(ix + 1, stat->resolution);           \
    float ## DIM v00 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix0), storage, channels);          \
    float ## DIM v10 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix1.x, ix0.y)), storage, channels); \
    float ## DIM v01 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, (int2)(ix0.x, ix1.y)), storage, channels); \
    float ## DIM v11 = _bufferIndexLinF ## DIM (                     \
        layer, _linearIndex(stat, ix1), storage, channels);          \
    return mix(mix(v00, v10, fx.x), mix(v01, v11, fx.x), fx.y);      \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
bufferSampleF ## DIM (const IMX_Layer* layer, float2 xy,             \
                     BorderType border, StorageType storage,         \
                     int channels)                                   \
{                                                                    \
    switch (border)                                                  \
    {                                                                \
    case (IMX_CONSTANT):                                             \
        return _bufferSampleF ## DIM ## _CN(                         \
            layer, xy, storage, channels);                           \
    case (IMX_CLAMP):                                                \
        return _bufferSampleF ## DIM ## _CL(                         \
            layer, xy, storage, channels);                           \
    case (IMX_MIRROR):                                               \
        return _bufferSampleF ## DIM ## _MR(                         \
            layer, xy, storage, channels);                           \
    case (IMX_WRAP): default:                                        \
        return _bufferSampleF ## DIM ## _WR(                         \
            layer, xy, storage, channels);                           \
    }                                                                \
}                                                                    \
                                                                     \
static float ## DIM                                                  \
bufferIndexF ## DIM (const IMX_Layer* layer, int2 xy,                \
                     BorderType border, StorageType storage,         \
                     int channels)                                   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
    {                                                                \
        switch (border)                                              \
        {                                                            \
        case (IMX_CONSTANT):                                         \
            return 0.0f;                                             \
        case (IMX_CLAMP):                                            \
            xy = clamp(xy, (int2)(0), res - 1);                      \
            break;                                                   \
        case (IMX_MIRROR):                                           \
            xy = _mirrorCoordinates(xy, res);                        \
            break;                                                   \
        case (IMX_WRAP): default:                                    \
            xy = _wrapCoordinates(xy, res);                          \
            break;                                                   \
        }                                                            \
    }                                                                \
    return _bufferIndexLinF ## DIM (                                 \
        layer, _linearIndex(layer->stat, xy), storage, channels);    \
}                                                                    \
                                                                     \
static void                                                          \
_setIndexF ## DIM(IMX_Layer *layer, int2 xy, float ## DIM val, StorageType storage, int channels)   \
{                                                                    \
    int2 res = layer->stat->resolution;                              \
    if (xy.x < 0 || xy.x >= res.x || xy.y < 0 || xy.y >= res.y)      \
        return;                                                      \
    _setIndexLinF ## DIM(layer, _linearIndex(layer->stat, xy), val, storage, channels); \
}                                                                    \
/**/

_IMPLEMENT_F(1)
_IMPLEMENT_F(2)
_IMPLEMENT_F(3)
_IMPLEMENT_F(4)
#undef _IMPLEMENT_F

static float2
applySTXform(STXform xform, float2 v)
{
    return v * xform.lo + xform.hi;
}

static float2
applySTXformInverse(STXform xform, float2 v)
{
    return (v - xform.hi) / xform.lo;
}

static float2
applySTXformVec(STXform xform, float2 v)
{
    return v * xform.lo;
}

static float2
applySTXformInverseVec(STXform xform, float2 v)
{
    return v / xform.lo;
}

static float2
bufferToImage(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXform(stat->buffer_to_image, v);
}

static float2
imageToBuffer(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXform(stat->image_to_buffer, v);
}

static float2
bufferToPixel(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXform(stat->buffer_to_pixel, v);
}

static float2
pixelToBuffer(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformInverse(stat->buffer_to_pixel, v);
}

static float2
bufferToTexture(global const IMX_Stat* restrict stat, float2 v)
{
    return (v + 0.5f) / (float2)(stat->resolution.x, stat->resolution.y);
}

static float2
textureToBuffer(global const IMX_Stat* restrict stat, float2 v)
{
    return v * ((float2)(stat->resolution.x, stat->resolution.y)) - 0.5f;
}

static float3
imageToWorld(global const IMX_Stat* restrict stat, float2 xy)
{
    float4      xyzw = 0;
    xyzw.xy = xy;
    xyzw.w = 1;
    xyzw = mat4vecmul(stat->image_to_world, xyzw);

    // No taper needed as we are on the plane.

    return xyzw.xyz;
}

static float3
image3ToWorld(global const IMX_Stat* restrict stat, float3 xyz)
{
    float4      xyzw = 0;
    xyzw.xyz = xyz;

    // We need to taper our image coordinates before
    // transforming back to world space.
    // Note z is not transformed here.
    if (stat->projection == IMX_PROJ_PERSPECTIVE)
    {
        float cameraz = stat->camera_image_pos.z;

        xyzw.xy *= (cameraz - xyzw.z) / cameraz;
    }

    xyzw.w = 1;
    xyzw = mat4vecmul(stat->image_to_world, xyzw);

    return xyzw.xyz;
}

static float2
worldToImage(global const IMX_Stat* restrict stat, float3 xyz)
{
    float4      xyzw = 0;
    xyzw.xyz = xyz;
    xyzw.w = 1;
    xyzw = mat4vecmul(stat->world_to_image, xyzw);

    // We are now in orthogonal coordinates that are "corrrect"
    // where the image plane is, so we need to account for the 
    // camera taper.
    if (stat->projection == IMX_PROJ_PERSPECTIVE)
    {
        float cameraz = stat->camera_image_pos.z;

        if (xyzw.z == cameraz)
            xyzw.xy = 0;
        else
            xyzw.xy *= cameraz / (cameraz - xyzw.z);
    }


    return xyzw.xy;
}

static float3
worldToImage3(global const IMX_Stat* restrict stat, float3 xyz)
{
    float4      xyzw = 0;
    xyzw.xyz = xyz;
    xyzw.w = 1;
    xyzw = mat4vecmul(stat->world_to_image, xyzw);

    // We are now in orthogonal coordinates that are "corrrect"
    // where the image plane is, so we need to account for the 
    // camera taper.
    if (stat->projection == IMX_PROJ_PERSPECTIVE)
    {
        float cameraz = stat->camera_image_pos.z;

        if (xyzw.z == cameraz)
            xyzw.xy = 0;
        else
            xyzw.xy *= cameraz / (cameraz - xyzw.z);
    }

    return xyzw.xyz;
}

static float2
bufferToImageVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformVec(stat->buffer_to_image, v);
}

static float2
imageToBufferVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformVec(stat->image_to_buffer, v);
}

static float2
bufferToPixelVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformVec(stat->buffer_to_pixel, v);
}

static float2
pixelToBufferVec(global const IMX_Stat* restrict stat, float2 v)
{
    return applySTXformInverseVec(stat->buffer_to_pixel, v);
}

static float2
bufferToTextureVec(global const IMX_Stat* restrict stat, float2 v)
{
    return (v) / (float2)(stat->resolution.x, stat->resolution.y);
}

static float2
textureToBufferVec(global const IMX_Stat* restrict stat, float2 v)
{
    return v * ((float2)(stat->resolution.x, stat->resolution.y));
}

static float3
imageToWorldVec(global const IMX_Stat* restrict stat, float2 xy)
{
    float3      xyz = 0;
    xyz.xy = xy;
    xyz = mat43vec3mul(stat->image_to_world, xyz);

    // No taper needed as we are on the plane.

    return xyz;
}

static float3
image3ToWorldVec(global const IMX_Stat* restrict stat, float3 xyz)
{
    // Directions are not consistent in tapered spaces,
    // so we just punt and go with the isometric case.
    xyz = mat43vec3mul(stat->image_to_world, xyz);

    return xyz;
}

static float2
worldToImageVec(global const IMX_Stat* restrict stat, float3 xyz)
{
    xyz = mat43vec3mul(stat->world_to_image, xyz);

    // Directions are not consistent in tapered spaces,
    // so we just punt and go with the isometric case.
    return xyz.xy;
}

static float3
worldToImage3Vec(global const IMX_Stat* restrict stat, float3 xyz)
{
    xyz = mat43vec3mul(stat->world_to_image, xyz);

    // Directions are not consistent in tapered spaces,
    // so we just punt and go with the isometric case.
    return xyz;
}

static float4
dCdxF4(const IMX_Layer* layer, float2 ixy,
       BorderType border, StorageType storage, int channels,
       const IMX_Layer* dst)
{
    float2 xy = imageToBuffer(layer->stat, ixy);
    float d = (dst->stat->buffer_to_image.x * layer->stat->image_to_buffer.x) / 2;
    float4 a = bufferSampleF4(layer, (float2)(xy.x + d, xy.y), border, storage, channels);
    float4 b = bufferSampleF4(layer, (float2)(xy.x - d, xy.y), border, storage, channels);
    return a - b;
}

static float4
dCdxF4aligned(const IMX_Layer* layer, int2 xy,
              BorderType border, StorageType storage, int channels)
{
    float4 a = bufferIndexF4(layer, (int2)(xy.x+1, xy.y), border, storage, channels);
    float4 b = bufferIndexF4(layer, (int2)(xy.x-1, xy.y), border, storage, channels);
    return (a-b) * 0.5f;
}

static float4
dCdyF4(const IMX_Layer* layer, float2 ixy,
       BorderType border, StorageType storage, int channels,
       const IMX_Layer* dst)
{
    float2 xy = imageToBuffer(layer->stat, ixy);
    float d = (dst->stat->buffer_to_image.y * layer->stat->image_to_buffer.y) / 2;
    float4 a = bufferSampleF4(layer, (float2)(xy.x, xy.y + d), border, storage, channels);
    float4 b = bufferSampleF4(layer, (float2)(xy.x, xy.y - d), border, storage, channels);
    return a - b;
}

static float4
dCdyF4aligned(const IMX_Layer* layer, int2 xy,
              BorderType border, StorageType storage, int channels)
{
    float4 a = bufferIndexF4(layer, (int2)(xy.x, xy.y+1), border, storage, channels);
    float4 b = bufferIndexF4(layer, (int2)(xy.x, xy.y-1), border, storage, channels);
    return (a-b) * 0.5f;
}

#endif

#ifndef __IMX_FILTER_H__
#define __IMX_FILTER_H__

// Only a single filter can be used in a snippet. To choose filter:
//
// Either define one of the following:
//  FILTER_POINT
//  FILTER_BILINEAR
//  FILTER_BOX (floating-point width box, sometimes called a 'tent')
//  FILTER_TRIANGLE
//  FILTER_CUBIC (bicubic interpolation if scale==1)
//  FILTER_MITCHELL (non-interpolating cubic)
//  FILTER_BSPLINE (smooth non-interpolating cubic)
// Or define these symbols, this example is equivalent to FILTER_TRIANGLE:
//  #define FILTER_SIZE 2
//  static __constant float samples[FILTER_SIZE] = { 1, 0 };
//  #define FILTER samples
//  #define FILTER_SUPPORT 1
//
// The following @-substitutions can then be used to sample areas, for float1-4 bindings
// F @name.bufferSampleRect(float2 center, float2 size) -> area in buffer coordinates
// F @name.bufferSampleRectClip(float2 center, float2 size) -> area in buffer coordinates
// F @name.imageSampleRect(float2 center, float2 size) -> area in image coordinates
// F @name.imageSampleRectClip(float2 center, float2 size) -> area in image coordinates

// #include "imx_filter_internal.h"
#endif
#if (!defined(FILTER) || !defined(FILTER_SIZE) || !defined(FILTER_SUPPORT)) && \
    !defined(FILTER_POINT) && !defined(FILTER_BILINEAR) && \
    !defined(FILTER_BOX) && !defined(FILTER_TRIANGLE) && \
    !defined(FILTER_CUBIC) && !defined(FILTER_MITCHELL) && \
    !defined(FILTER_BSPLINE)
#define FILTER_BOX
#endif

#if defined(FILTER) && defined(FILTER_SIZE) && defined(FILTER_SUPPORT)
// this is so snippet can define it's own filter before including this

#elif defined(FILTER_POINT)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    // Make .5 round toward -infinity so a translate by .5 does not throw away every other pixel
    return bufferIndexF4(layer, convert_int2_sat_rtn(xy + 0.5f), border, storage, channels);
}

#elif defined(FILTER_BILINEAR)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    return bufferSampleF4(layer, xy, border, storage, channels);
}

#elif defined(FILTER_BOX)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    xy = clamp(xy, -1e8f, 1e8f);
    dxy = clamp(dxy, 1.0f, 67.0f);
    float2 r = (dxy + 1.0f) / 2;
    int2 ia; float2 fa;
    _splitCoordinates(r - xy, &ia, &fa);
    ia = -ia;
    int2 ib; float2 fb;
    _splitCoordinates(xy + r, &ib, &fb);
    float4 sum = 0;
    int2 ixy;
    float w = fa.y;
    for (ixy.y = ia.y; ixy.y <= ib.y;)
    {
        ixy.x = ia.x;
        float4 sum1 = fa.x * bufferIndexF4(layer, ixy, border, storage, channels);
        for (ixy.x++; ixy.x < ib.x; ixy.x++)
            sum1 += bufferIndexF4(layer, ixy, border, storage, channels);
        sum1 += fb.x * bufferIndexF4(layer, ixy, border, storage, channels);
        sum += w * sum1;
        w = (++ixy.y) < ib.y ? 1.0f : fb.y;
    }
    return sum / (dxy.x * dxy.y);
}

#elif defined(FILTER_TRIANGLE)
static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    float2 r = clamp(dxy, 1.0f, 1.0e8f);
    int2 ia = convert_int2_sat_rtp(xy - r);
    int2 ib = convert_int2_sat_rtp(xy + r);
    float4 sum = 0;
    float div = 0;
    int2 ixy;
    int2 inc = (ib - ia) / 61 + 1;
    for (ixy.y = ia.y; ixy.y < ib.y; ixy.y += inc.y)
    {
        float4 sum1 = 0;
        float div1 = 0;
        for (ixy.x = ia.x; ixy.x < ib.x; ixy.x += inc.x)
        {
            float w = 1 - fabs(ixy.x - xy.x) / r.x;
            sum1 += w * bufferIndexF4(layer, ixy, border, storage, channels);
            div1 += w;
        }
        float w = 1 - fabs(ixy.y - xy.y) / r.y;
        sum += w * sum1;
        div += w * div1;
    }
    return sum / div;
}

#elif defined(FILTER_CUBIC)
#define FILTER_SIZE 25
static __constant float Cubic[FILTER_SIZE] = { // mitchell(25, 0, 0.5)
    6, 5.9010415, 5.625, 5.203125, 4.6666665, 4.046875, 3.375, 2.6822917,
    2, 1.359375, 0.7916667, 0.328125, 0, -0.21006945, -0.3472222, -0.421875,
    -0.44444445, -0.4253472, -0.375, -0.30381945, -0.22222222, -0.140625, -0.06944445, -0.019097222,
    0};
#define FILTER Cubic
#define FILTER_SUPPORT 2

#elif defined(FILTER_MITCHELL)
#define FILTER_SIZE 25
static __constant float Mitchell[FILTER_SIZE] = { // mitchell(25, 1/3.0, 1/3.0)
    5.3333335, 5.2540507, 5.0324073, 4.6927085, 4.259259, 3.7563658, 3.2083333, 2.6394675,
    2.074074, 1.5364584, 1.050926, 0.6417824, 0.33333334, 0.116705246, -0.038580246, -0.140625,
    -0.19753087, -0.21739969, -0.20833333, -0.17843364, -0.13580246, -0.088541664, -0.044753086, -0.01253858,
    0}; //-1.7763568e-15};
#define FILTER Mitchell
#define FILTER_SUPPORT 2

#elif defined(FILTER_BSPLINE)
#define FILTER_SIZE 25
static __constant float BSpline[FILTER_SIZE] = { // mitchell(25, 1, 0)
    4, 3.9600694, 3.8472223, 3.671875, 3.4444444, 3.1753473, 2.875, 2.5538194,
    2.2222223, 1.890625, 1.5694444, 1.2690972, 1, 0.7702546, 0.5787037, 0.421875,
    0.2962963, 0.19849537, 0.125, 0.07233796, 0.037037037, 0.015625, 0.0046296297, 0.0005787037,
    0};
#define FILTER BSpline
#define FILTER_SUPPORT 2

#endif

#if defined(FILTER)

static float
sampleLookup(constant float * in, float pos)
{
    float flr;
    float t = fract(pos, &flr);
    int flooridx = convert_int(flr);
    int ceilidx = flooridx+1;
    ceilidx = min(ceilidx, FILTER_SIZE-1);
    return mix(in[flooridx], in[ceilidx], t);
}

static float4
bufferSampleRectF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                  BorderType border, StorageType storage, int channels)
{
    float2 r = min(max(dxy, 1.0f) * FILTER_SUPPORT, 1.0e8f);
    int2 ia = convert_int2_sat_rtp(xy - r);
    int2 ib = convert_int2_sat_rtp(xy + r);
    float2 k = (FILTER_SIZE - 1) / r;
    float4 sum = 0;
    float div = 0;
    int2 ixy;
    int2 inc = (ib - ia) / 61 + 1;
    for (ixy.y = ia.y; ixy.y < ib.y; ixy.y += inc.y)
    {
        float4 sum1 = 0;
        float div1 = 0;
        for (ixy.x = ia.x; ixy.x < ib.x; ixy.x += inc.x)
        {
            float w = sampleLookup(FILTER, fabs(ixy.x - xy.x) * k.x);
            sum1 += w * bufferIndexF4(layer, ixy, border, storage, channels);
            div1 += w;
        }
        float w = sampleLookup(FILTER, fabs(ixy.y - xy.y) * k.y);
        sum += w * sum1;
        div += w * div1;
    }
    return sum / div;
}
#endif

// Intersects the incoming sampling window with the layer's footprint and
// returns the filtered value in that rectangle. (Thus, this should never read
// outside pixels.)
static float4
bufferSampleRectClipF4(const IMX_Layer* layer, float2 xy, float2 dxy,
                       StorageType storage, int channels)
{
#if defined(FILTER_POINT)
    return bufferSampleRectF4(layer, xy, dxy, IMX_CONSTANT, storage, channels);
#else
    float2 wh = convert_float2(layer->stat->resolution);
    dxy *= 0.5f;
    float2 xy0 = max(xy - dxy, -0.5f);
    float2 xy1 = min(xy + dxy, wh - 0.5f);
    if (any(xy0 >= xy1))
        return 0.0f;
    return bufferSampleRectF4(layer, (xy1 + xy0) * 0.5f, xy1 - xy0, IMX_CLAMP,
                              storage, channels);
#endif
}

static float4
constImageSampleRectClip(float2 xy, float2 dxy, float4 defval)
{
    // Clip to image space; if there is any intersection, return defval;
    // otherwise, return 0 since we're fully outside the image.
    float2 xy0 = max(xy - dxy, -1.0f);
    float2 xy1 = min(xy + dxy, 1.0f);
    return any(xy0 >= xy1) ? 0.0f : defval;
}

// convert derivatives (or a parallelogram) to nearest ortho rectangle
static float2
wh_from_dP(float2 dPdx, float2 dPdy)
{
    return hypot(dPdx, dPdy);
}

/* Python to generate the filters
def mitchell(n, B, C):
    print("[%d] = {"%n, end='');
    for i in range(0,n):
        if i: print(",", end='')
        if not(i%8): print("\n   ", end='')
        x = (2.0*i)/(n-1)
        if (x < 1):
            v = ((12-9*B-6*C)*x-18+12*B+6*C)*x*x+6-2*B;
        else:
            v = (((-B-6*C)*x+6*B+30*C)*x-12*B-48*C)*x+8*B+24*C;
        print(" %.6g"%v, end='')
    print("};")

mitchell(25, 1/3.0, 1/3.0)
*/
#define AT_elemnum      _bound_idx
#define AT_ix   _bound_gidx
#define AT_iy   _bound_gidy
#define AT_ixy  (int2)(_bound_gidx, _bound_gidy)
#define AT_res  (_RUNOVER_LAYER.stat->resolution)
#define AT_xres (_RUNOVER_LAYER.stat->resolution.x)
#define AT_yres (_RUNOVER_LAYER.stat->resolution.y)
#define AT_P_image      _bound_P_image
#define AT_P_pixel      _bound_P_pixel
#define AT_P_texture    _bound_P_texture
#define AT_P_world      (imageToWorld(_RUNOVER_LAYER.stat, _bound_P_image))
#define AT_P    AT_P_image
#define AT_dPdx_image   ((float2)(_RUNOVER_LAYER.stat->buffer_to_image.x,0))
#define AT_dPdx_pixel   ((float2)(_RUNOVER_LAYER.stat->buffer_to_pixel.x,0))
#define AT_dPdx_texture ((float2)(1.0f/(float)_RUNOVER_LAYER.stat->resolution.x,0))
#define AT_dPdx AT_dPdx_image
#define AT_dPdy_image   ((float2)(0, _RUNOVER_LAYER.stat->buffer_to_image.y))
#define AT_dPdy_pixel   ((float2)(0, _RUNOVER_LAYER.stat->buffer_to_pixel.y))
#define AT_dPdy_texture ((float2)(0, 1.0f/(float)_RUNOVER_LAYER.stat->resolution.y))
#define AT_dPdy AT_dPdy_image
#define AT_dPdxy_image (_RUNOVER_LAYER.stat->buffer_to_image.xy)
#define AT_dPdxy_pixel (_RUNOVER_LAYER.stat->buffer_to_pixel.xy)
#define AT_dPdxy_texture ((float2)(1.0f/(float)_RUNOVER_LAYER.stat->resolution.x,1.0f/(float)_RUNOVER_LAYER.stat->resolution.y))
#define AT_dPdxy AT_dPdxy_image
#define AT_tilesize     _bound_tilesize
#define AT_Time _bound_time
#define AT_iDate        _bound_iDate
#define AT_iFrame       _bound_iFrame
#define AT_iFrameRate   _bound_iFrameRate
#define AT_iMouse       _bound_iMouse
#ifdef HAS_size_ref
#define AT_size_ref_data        _bound_size_ref
#else
#define AT_size_ref_data        0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bound       1
#else
#define AT_size_ref_bound       0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_stat        ((global IMX_Stat * restrict) _bound_size_ref_stat_void)
#else
#define AT_size_ref_stat        0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_layer       &_bound_size_ref_layer
#else
#define AT_size_ref_layer       0
#endif
#ifdef HAS_size_ref
#define AT_size_ref_border      _bound_size_ref_border
#else
#define AT_size_ref_border      IMX_WRAP
#endif
#ifdef HAS_size_ref
#define AT_size_ref_storage     _bound_size_ref_storage
#else
#define AT_size_ref_storage     FLOAT32
#endif
#ifdef HAS_size_ref
#define AT_size_ref_channels    _bound_size_ref_channels
#else
#define AT_size_ref_channels    4
#endif
#ifdef HAS_size_ref
#define AT_size_ref_tuplesize   _bound_size_ref_channels
#else
#define AT_size_ref_tuplesize   4
#endif
#ifdef HAS_size_ref
#define AT_size_ref_xres        _bound_size_ref_layer.stat->resolution.x
#else
#define AT_size_ref_xres        1
#endif
#ifdef HAS_size_ref
#define AT_size_ref_yres        _bound_size_ref_layer.stat->resolution.y
#else
#define AT_size_ref_yres        1
#endif
#ifdef HAS_size_ref
#define AT_size_ref_res convert_float2(_bound_size_ref_layer.stat->resolution)
#else
#define AT_size_ref_res (float2)(1)
#endif
#define CONSTANT1(s) CONSTANT_ ## s
#define CONSTANT_(s) CONSTANT1(s)
#ifdef CONSTANT_size_ref
#define size_ref_args2 CONSTANT_(_bound_size_ref_storage), _bound_size_ref_channels
#else
#define size_ref_args2 _bound_size_ref_storage, _bound_size_ref_channels
#endif
#define size_ref_args3 _bound_size_ref_border, size_ref_args2
#ifdef HAS_size_ref
#define AT_size_ref_bufferIndex(_xy_)   bufferIndexF4(&_bound_size_ref_layer, _xy_, size_ref_args3)
#else
#define AT_size_ref_bufferIndex(_xy_)   _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferSample(_xy_)  bufferSampleF4(&_bound_size_ref_layer, _xy_, size_ref_args3)
#else
#define AT_size_ref_bufferSample(_xy_)  _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageNearest(_xy_)  bufferIndexF4(&_bound_size_ref_layer, convert_int2_sat_rtn(imageToBuffer(AT_size_ref_stat, _xy_) + 0.5f), size_ref_args3)
#else
#define AT_size_ref_imageNearest(_xy_)  _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageSample(_xy_)   bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, _xy_), size_ref_args3)
#else
#define AT_size_ref_imageSample(_xy_)   _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldNearest(_xyz_) bufferIndexF4(&_bound_size_ref_layer, convert_int2_sat_rtn(imageToBuffer(AT_size_ref_stat, worldToImage(AT_size_ref_stat, _xyz_)) + 0.5f), size_ref_args3)
#else
#define AT_size_ref_worldNearest(_xyz_) _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldSample(_xyz_)  bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, worldToImage(AT_size_ref_stat, _xyz_)), size_ref_args3)
#else
#define AT_size_ref_worldSample(_xyz_)  _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureNearest(_xy_)        bufferIndexF4(&_bound_size_ref_layer, convert_int2_sat_rtn(textureToBuffer(AT_size_ref_stat, _xy_) + 0.5f), size_ref_args3)
#else
#define AT_size_ref_textureNearest(_xy_)        _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureSample(_xy_) bufferSampleF4(&_bound_size_ref_layer, textureToBuffer(AT_size_ref_stat, _xy_), size_ref_args3)
#else
#define AT_size_ref_textureSample(_xy_) _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_1(_xy_)     bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, _xy_), size_ref_args3)
#else
#define AT_size_ref_1(_xy_)     _bound_size_ref
#endif
#ifdef HAS_size_ref
#ifdef ALIGNED_size_ref
#define AT_size_ref     _bufferIndexLinF4(&_bound_size_ref_layer, _bound_idx, size_ref_args2)
#else
#define AT_size_ref     bufferSampleF4(&_bound_size_ref_layer, imageToBuffer(AT_size_ref_stat, _bound_P_image), size_ref_args3)
#endif
#else
#define AT_size_ref     _bound_size_ref
#endif
#ifdef HAS_size_ref
#ifdef ALIGNED_size_ref
#define AT_size_ref_dCdx        dCdxF4aligned(&_bound_size_ref_layer, (int2)(_bound_gidx, _bound_gidy), size_ref_args3)
#else
#define AT_size_ref_dCdx        dCdxF4(&_bound_size_ref_layer, _bound_P_image, size_ref_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_size_ref_dCdx        ((float4)0)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_dCdx_1(_xy_)        dCdxF4(&_bound_size_ref_layer, _xy_, size_ref_args3, &_RUNOVER_LAYER)
#else
#define AT_size_ref_dCdx_1(_xy_)        ((float4)0)
#endif
#ifdef HAS_size_ref
#ifdef ALIGNED_size_ref
#define AT_size_ref_dCdy        dCdyF4aligned(&_bound_size_ref_layer, (int2)(_bound_gidx, _bound_gidy), size_ref_args3)
#else
#define AT_size_ref_dCdy        dCdyF4(&_bound_size_ref_layer, _bound_P_image, size_ref_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_size_ref_dCdy        ((float4)0)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_dCdy_1(_xy_)        dCdyF4(&_bound_size_ref_layer, _xy_, size_ref_args3, &_RUNOVER_LAYER)
#else
#define AT_size_ref_dCdy_1(_xy_)        ((float4)0)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferSampleRect(_xy_, _dxy_)       bufferSampleRectF4(&_bound_size_ref_layer, _xy_, _dxy_, size_ref_args3)
#else
#define AT_size_ref_bufferSampleRect(_xy_, _dxy_)       _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferSampleRectClip(_xy_, _dxy_)   bufferSampleRectClipF4(&_bound_size_ref_layer, _xy_, _dxy_, size_ref_args2)
#else
#define AT_size_ref_bufferSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(bufferToImage(AT_size_ref_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y)), _bound_size_ref)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageSampleRect(_xy_, _dxy_)        AT_size_ref_bufferSampleRect(imageToBuffer(AT_size_ref_stat, _xy_), AT_size_ref_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_size_ref_imageSampleRect(_xy_, _dxy_)        _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageSampleRectClip(_xy_, _dxy_)    AT_size_ref_bufferSampleRectClip(imageToBuffer(AT_size_ref_stat, _xy_), AT_size_ref_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_size_ref_imageSampleRectClip(_xy_, _dxy_)    constImageSampleRectClip(_xy_, _dxy_, _bound_size_ref)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureSampleRect(_xy_, _dxy_)      AT_size_ref_bufferSampleRect(textureToBuffer(AT_size_ref_stat, _xy_), (float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y) * (_dxy_))
#else
#define AT_size_ref_textureSampleRect(_xy_, _dxy_)      _bound_size_ref
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureSampleRectClip(_xy_, _dxy_)  AT_size_ref_bufferSampleRectClip(textureToBuffer(AT_size_ref_stat, _xy_), (float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y) * (_dxy_))
#else
#define AT_size_ref_textureSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_size_ref_stat, textureToBuffer(AT_size_ref_stat, _xy_)), _dxy_ * ((float2)(AT_size_ref_stat->resolution.x, AT_size_ref_stat->resolution.y)) * AT_size_ref_stat->buffer_to_image.lo, _bound_size_ref)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferToImage(_xy_) (bufferToImage(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_bufferToImage(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageToBuffer(_xy_) (imageToBuffer(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_imageToBuffer(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferToPixel(_xy_) (bufferToPixel(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_bufferToPixel(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_pixelToBuffer(_xy_) (pixelToBuffer(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_pixelToBuffer(_xy_) (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_bufferToTexture(_xy_)       (bufferToTexture(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_bufferToTexture(_xy_)       (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_textureToBuffer(_xy_)       (textureToBuffer(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_textureToBuffer(_xy_)       (_xy_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_imageToWorld(_xy_)  (imageToWorld(AT_size_ref_stat, _xy_))
#else
#define AT_size_ref_imageToWorld(_xy_)  ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_size_ref
#define AT_size_ref_image3ToWorld(_xyz_)        (image3ToWorld(AT_size_ref_stat, _xyz_))
#else
#define AT_size_ref_image3ToWorld(_xyz_)        (_xyz_)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldToImage(_xyz_) (worldToImage(AT_size_ref_stat, _xyz_))
#else
#define AT_size_ref_worldToImage(_xyz_) ((_xyz_).xy)
#endif
#ifdef HAS_size_ref
#define AT_size_ref_worldToImage3(_xyz_)        (worldToImage3(AT_size_ref_stat, _xyz_))
#else
#define AT_size_ref_worldToImage3(_xyz_)        (_xyz_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_data       _bound_fragCoord
#else
#define AT_fragCoord_data       0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bound      1
#else
#define AT_fragCoord_bound      0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_stat       ((global IMX_Stat * restrict) _bound_fragCoord_stat_void)
#else
#define AT_fragCoord_stat       0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_layer      &_bound_fragCoord_layer
#else
#define AT_fragCoord_layer      0
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_border     _bound_fragCoord_border
#else
#define AT_fragCoord_border     IMX_WRAP
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_storage    _bound_fragCoord_storage
#else
#define AT_fragCoord_storage    FLOAT32
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_channels   _bound_fragCoord_channels
#else
#define AT_fragCoord_channels   4
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_tuplesize  _bound_fragCoord_channels
#else
#define AT_fragCoord_tuplesize  4
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_xres       _bound_fragCoord_layer.stat->resolution.x
#else
#define AT_fragCoord_xres       1
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_yres       _bound_fragCoord_layer.stat->resolution.y
#else
#define AT_fragCoord_yres       1
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_res        convert_float2(_bound_fragCoord_layer.stat->resolution)
#else
#define AT_fragCoord_res        (float2)(1)
#endif
#ifdef CONSTANT_fragCoord
#define fragCoord_args2 CONSTANT_(_bound_fragCoord_storage), _bound_fragCoord_channels
#else
#define fragCoord_args2 _bound_fragCoord_storage, _bound_fragCoord_channels
#endif
#define fragCoord_args3 _bound_fragCoord_border, fragCoord_args2
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferIndex(_xy_)  bufferIndexF2(&_bound_fragCoord_layer, _xy_, fragCoord_args3)
#else
#define AT_fragCoord_bufferIndex(_xy_)  _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferSample(_xy_) bufferSampleF2(&_bound_fragCoord_layer, _xy_, fragCoord_args3)
#else
#define AT_fragCoord_bufferSample(_xy_) _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageNearest(_xy_) bufferIndexF2(&_bound_fragCoord_layer, convert_int2_sat_rtn(imageToBuffer(AT_fragCoord_stat, _xy_) + 0.5f), fragCoord_args3)
#else
#define AT_fragCoord_imageNearest(_xy_) _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageSample(_xy_)  bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, _xy_), fragCoord_args3)
#else
#define AT_fragCoord_imageSample(_xy_)  _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldNearest(_xyz_)        bufferIndexF2(&_bound_fragCoord_layer, convert_int2_sat_rtn(imageToBuffer(AT_fragCoord_stat, worldToImage(AT_fragCoord_stat, _xyz_)) + 0.5f), fragCoord_args3)
#else
#define AT_fragCoord_worldNearest(_xyz_)        _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldSample(_xyz_) bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, worldToImage(AT_fragCoord_stat, _xyz_)), fragCoord_args3)
#else
#define AT_fragCoord_worldSample(_xyz_) _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureNearest(_xy_)       bufferIndexF2(&_bound_fragCoord_layer, convert_int2_sat_rtn(textureToBuffer(AT_fragCoord_stat, _xy_) + 0.5f), fragCoord_args3)
#else
#define AT_fragCoord_textureNearest(_xy_)       _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureSample(_xy_)        bufferSampleF2(&_bound_fragCoord_layer, textureToBuffer(AT_fragCoord_stat, _xy_), fragCoord_args3)
#else
#define AT_fragCoord_textureSample(_xy_)        _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_1(_xy_)    bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, _xy_), fragCoord_args3)
#else
#define AT_fragCoord_1(_xy_)    _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#ifdef ALIGNED_fragCoord
#define AT_fragCoord    _bufferIndexLinF2(&_bound_fragCoord_layer, _bound_idx, fragCoord_args2)
#else
#define AT_fragCoord    bufferSampleF2(&_bound_fragCoord_layer, imageToBuffer(AT_fragCoord_stat, _bound_P_image), fragCoord_args3)
#endif
#else
#define AT_fragCoord    _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#ifdef ALIGNED_fragCoord
#define AT_fragCoord_dCdx       dCdxF4aligned(&_bound_fragCoord_layer, (int2)(_bound_gidx, _bound_gidy), fragCoord_args3).xy
#else
#define AT_fragCoord_dCdx       dCdxF4(&_bound_fragCoord_layer, _bound_P_image, fragCoord_args3, &_RUNOVER_LAYER).xy
#endif
#else
#define AT_fragCoord_dCdx       ((float2)0)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_dCdx_1(_xy_)       dCdxF4(&_bound_fragCoord_layer, _xy_, fragCoord_args3, &_RUNOVER_LAYER).xy
#else
#define AT_fragCoord_dCdx_1(_xy_)       ((float2)0)
#endif
#ifdef HAS_fragCoord
#ifdef ALIGNED_fragCoord
#define AT_fragCoord_dCdy       dCdyF4aligned(&_bound_fragCoord_layer, (int2)(_bound_gidx, _bound_gidy), fragCoord_args3).xy
#else
#define AT_fragCoord_dCdy       dCdyF4(&_bound_fragCoord_layer, _bound_P_image, fragCoord_args3, &_RUNOVER_LAYER).xy
#endif
#else
#define AT_fragCoord_dCdy       ((float2)0)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_dCdy_1(_xy_)       dCdyF4(&_bound_fragCoord_layer, _xy_, fragCoord_args3, &_RUNOVER_LAYER).xy
#else
#define AT_fragCoord_dCdy_1(_xy_)       ((float2)0)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_fragCoord_layer, _xy_, _dxy_, fragCoord_args3).xy
#else
#define AT_fragCoord_bufferSampleRect(_xy_, _dxy_)      _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_fragCoord_layer, _xy_, _dxy_, fragCoord_args2).xy
#else
#define AT_fragCoord_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_fragCoord_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y)), _bound_fragCoord).xy
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageSampleRect(_xy_, _dxy_)       AT_fragCoord_bufferSampleRect(imageToBuffer(AT_fragCoord_stat, _xy_), AT_fragCoord_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_fragCoord_imageSampleRect(_xy_, _dxy_)       _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageSampleRectClip(_xy_, _dxy_)   AT_fragCoord_bufferSampleRectClip(imageToBuffer(AT_fragCoord_stat, _xy_), AT_fragCoord_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_fragCoord_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_fragCoord).xy
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureSampleRect(_xy_, _dxy_)     AT_fragCoord_bufferSampleRect(textureToBuffer(AT_fragCoord_stat, _xy_), (float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y) * (_dxy_))
#else
#define AT_fragCoord_textureSampleRect(_xy_, _dxy_)     _bound_fragCoord
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureSampleRectClip(_xy_, _dxy_) AT_fragCoord_bufferSampleRectClip(textureToBuffer(AT_fragCoord_stat, _xy_), (float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y) * (_dxy_))
#else
#define AT_fragCoord_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_fragCoord_stat, textureToBuffer(AT_fragCoord_stat, _xy_)), _dxy_ * ((float2)(AT_fragCoord_stat->resolution.x, AT_fragCoord_stat->resolution.y)) * AT_fragCoord_stat->buffer_to_image.lo, _bound_fragCoord).xy
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferToImage(_xy_)        (bufferToImage(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageToBuffer(_xy_)        (imageToBuffer(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferToPixel(_xy_)        (bufferToPixel(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_pixelToBuffer(_xy_)        (pixelToBuffer(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_bufferToTexture(_xy_)      (bufferToTexture(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_textureToBuffer(_xy_)      (textureToBuffer(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_imageToWorld(_xy_) (imageToWorld(AT_fragCoord_stat, _xy_))
#else
#define AT_fragCoord_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_image3ToWorld(_xyz_)       (image3ToWorld(AT_fragCoord_stat, _xyz_))
#else
#define AT_fragCoord_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldToImage(_xyz_)        (worldToImage(AT_fragCoord_stat, _xyz_))
#else
#define AT_fragCoord_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_fragCoord
#define AT_fragCoord_worldToImage3(_xyz_)       (worldToImage3(AT_fragCoord_stat, _xyz_))
#else
#define AT_fragCoord_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_data       _bound_iChannel0
#else
#define AT_iChannel0_data       0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bound      1
#else
#define AT_iChannel0_bound      0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_stat       ((global IMX_Stat * restrict) _bound_iChannel0_stat_void)
#else
#define AT_iChannel0_stat       0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_layer      &_bound_iChannel0_layer
#else
#define AT_iChannel0_layer      0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_border     _bound_iChannel0_border
#else
#define AT_iChannel0_border     IMX_WRAP
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_storage    _bound_iChannel0_storage
#else
#define AT_iChannel0_storage    FLOAT32
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_channels   _bound_iChannel0_channels
#else
#define AT_iChannel0_channels   4
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_tuplesize  _bound_iChannel0_channels
#else
#define AT_iChannel0_tuplesize  4
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_xres       _bound_iChannel0_layer.stat->resolution.x
#else
#define AT_iChannel0_xres       1
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_yres       _bound_iChannel0_layer.stat->resolution.y
#else
#define AT_iChannel0_yres       1
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_res        convert_float2(_bound_iChannel0_layer.stat->resolution)
#else
#define AT_iChannel0_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel0
#define iChannel0_args2 CONSTANT_(_bound_iChannel0_storage), _bound_iChannel0_channels
#else
#define iChannel0_args2 _bound_iChannel0_storage, _bound_iChannel0_channels
#endif
#define iChannel0_args3 _bound_iChannel0_border, iChannel0_args2
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3)
#else
#define AT_iChannel0_bufferIndex(_xy_)  _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3)
#else
#define AT_iChannel0_bufferSample(_xy_) _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel0_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel0_stat, _xy_) + 0.5f), iChannel0_args3)
#else
#define AT_iChannel0_imageNearest(_xy_) _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, _xy_), iChannel0_args3)
#else
#define AT_iChannel0_imageSample(_xy_)  _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel0_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel0_stat, worldToImage(AT_iChannel0_stat, _xyz_)) + 0.5f), iChannel0_args3)
#else
#define AT_iChannel0_worldNearest(_xyz_)        _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, worldToImage(AT_iChannel0_stat, _xyz_)), iChannel0_args3)
#else
#define AT_iChannel0_worldSample(_xyz_) _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel0_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel0_stat, _xy_) + 0.5f), iChannel0_args3)
#else
#define AT_iChannel0_textureNearest(_xy_)       _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel0_layer, textureToBuffer(AT_iChannel0_stat, _xy_), iChannel0_args3)
#else
#define AT_iChannel0_textureSample(_xy_)        _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_1(_xy_)    bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, _xy_), iChannel0_args3)
#else
#define AT_iChannel0_1(_xy_)    _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#ifdef ALIGNED_iChannel0
#define AT_iChannel0    _bufferIndexLinF4(&_bound_iChannel0_layer, _bound_idx, iChannel0_args2)
#else
#define AT_iChannel0    bufferSampleF4(&_bound_iChannel0_layer, imageToBuffer(AT_iChannel0_stat, _bound_P_image), iChannel0_args3)
#endif
#else
#define AT_iChannel0    _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#ifdef ALIGNED_iChannel0
#define AT_iChannel0_dCdx       dCdxF4aligned(&_bound_iChannel0_layer, (int2)(_bound_gidx, _bound_gidy), iChannel0_args3)
#else
#define AT_iChannel0_dCdx       dCdxF4(&_bound_iChannel0_layer, _bound_P_image, iChannel0_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel0_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel0_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel0
#ifdef ALIGNED_iChannel0
#define AT_iChannel0_dCdy       dCdyF4aligned(&_bound_iChannel0_layer, (int2)(_bound_gidx, _bound_gidy), iChannel0_args3)
#else
#define AT_iChannel0_dCdy       dCdyF4(&_bound_iChannel0_layer, _bound_P_image, iChannel0_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel0_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel0_layer, _xy_, iChannel0_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel0_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel0_layer, _xy_, _dxy_, iChannel0_args3)
#else
#define AT_iChannel0_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel0_layer, _xy_, _dxy_, iChannel0_args2)
#else
#define AT_iChannel0_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel0_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y)), _bound_iChannel0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageSampleRect(_xy_, _dxy_)       AT_iChannel0_bufferSampleRect(imageToBuffer(AT_iChannel0_stat, _xy_), AT_iChannel0_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel0_imageSampleRect(_xy_, _dxy_)       _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel0_bufferSampleRectClip(imageToBuffer(AT_iChannel0_stat, _xy_), AT_iChannel0_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel0_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureSampleRect(_xy_, _dxy_)     AT_iChannel0_bufferSampleRect(textureToBuffer(AT_iChannel0_stat, _xy_), (float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel0_textureSampleRect(_xy_, _dxy_)     _bound_iChannel0
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureSampleRectClip(_xy_, _dxy_) AT_iChannel0_bufferSampleRectClip(textureToBuffer(AT_iChannel0_stat, _xy_), (float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel0_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel0_stat, textureToBuffer(AT_iChannel0_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel0_stat->resolution.x, AT_iChannel0_stat->resolution.y)) * AT_iChannel0_stat->buffer_to_image.lo, _bound_iChannel0)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferToImage(_xy_)        (bufferToImage(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_imageToWorld(_xy_) (imageToWorld(AT_iChannel0_stat, _xy_))
#else
#define AT_iChannel0_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel0_stat, _xyz_))
#else
#define AT_iChannel0_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldToImage(_xyz_)        (worldToImage(AT_iChannel0_stat, _xyz_))
#else
#define AT_iChannel0_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel0
#define AT_iChannel0_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel0_stat, _xyz_))
#else
#define AT_iChannel0_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_data       _bound_iChannel1
#else
#define AT_iChannel1_data       0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bound      1
#else
#define AT_iChannel1_bound      0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_stat       ((global IMX_Stat * restrict) _bound_iChannel1_stat_void)
#else
#define AT_iChannel1_stat       0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_layer      &_bound_iChannel1_layer
#else
#define AT_iChannel1_layer      0
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_border     _bound_iChannel1_border
#else
#define AT_iChannel1_border     IMX_WRAP
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_storage    _bound_iChannel1_storage
#else
#define AT_iChannel1_storage    FLOAT32
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_channels   _bound_iChannel1_channels
#else
#define AT_iChannel1_channels   4
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_tuplesize  _bound_iChannel1_channels
#else
#define AT_iChannel1_tuplesize  4
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_xres       _bound_iChannel1_layer.stat->resolution.x
#else
#define AT_iChannel1_xres       1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_yres       _bound_iChannel1_layer.stat->resolution.y
#else
#define AT_iChannel1_yres       1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_res        convert_float2(_bound_iChannel1_layer.stat->resolution)
#else
#define AT_iChannel1_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel1
#define iChannel1_args2 CONSTANT_(_bound_iChannel1_storage), _bound_iChannel1_channels
#else
#define iChannel1_args2 _bound_iChannel1_storage, _bound_iChannel1_channels
#endif
#define iChannel1_args3 _bound_iChannel1_border, iChannel1_args2
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3)
#else
#define AT_iChannel1_bufferIndex(_xy_)  _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3)
#else
#define AT_iChannel1_bufferSample(_xy_) _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel1_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel1_stat, _xy_) + 0.5f), iChannel1_args3)
#else
#define AT_iChannel1_imageNearest(_xy_) _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, _xy_), iChannel1_args3)
#else
#define AT_iChannel1_imageSample(_xy_)  _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel1_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel1_stat, worldToImage(AT_iChannel1_stat, _xyz_)) + 0.5f), iChannel1_args3)
#else
#define AT_iChannel1_worldNearest(_xyz_)        _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, worldToImage(AT_iChannel1_stat, _xyz_)), iChannel1_args3)
#else
#define AT_iChannel1_worldSample(_xyz_) _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel1_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel1_stat, _xy_) + 0.5f), iChannel1_args3)
#else
#define AT_iChannel1_textureNearest(_xy_)       _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel1_layer, textureToBuffer(AT_iChannel1_stat, _xy_), iChannel1_args3)
#else
#define AT_iChannel1_textureSample(_xy_)        _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_1(_xy_)    bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, _xy_), iChannel1_args3)
#else
#define AT_iChannel1_1(_xy_)    _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#ifdef ALIGNED_iChannel1
#define AT_iChannel1    _bufferIndexLinF4(&_bound_iChannel1_layer, _bound_idx, iChannel1_args2)
#else
#define AT_iChannel1    bufferSampleF4(&_bound_iChannel1_layer, imageToBuffer(AT_iChannel1_stat, _bound_P_image), iChannel1_args3)
#endif
#else
#define AT_iChannel1    _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#ifdef ALIGNED_iChannel1
#define AT_iChannel1_dCdx       dCdxF4aligned(&_bound_iChannel1_layer, (int2)(_bound_gidx, _bound_gidy), iChannel1_args3)
#else
#define AT_iChannel1_dCdx       dCdxF4(&_bound_iChannel1_layer, _bound_P_image, iChannel1_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel1_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel1_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel1
#ifdef ALIGNED_iChannel1
#define AT_iChannel1_dCdy       dCdyF4aligned(&_bound_iChannel1_layer, (int2)(_bound_gidx, _bound_gidy), iChannel1_args3)
#else
#define AT_iChannel1_dCdy       dCdyF4(&_bound_iChannel1_layer, _bound_P_image, iChannel1_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel1_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel1_layer, _xy_, iChannel1_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel1_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel1_layer, _xy_, _dxy_, iChannel1_args3)
#else
#define AT_iChannel1_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel1_layer, _xy_, _dxy_, iChannel1_args2)
#else
#define AT_iChannel1_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel1_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y)), _bound_iChannel1)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageSampleRect(_xy_, _dxy_)       AT_iChannel1_bufferSampleRect(imageToBuffer(AT_iChannel1_stat, _xy_), AT_iChannel1_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel1_imageSampleRect(_xy_, _dxy_)       _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel1_bufferSampleRectClip(imageToBuffer(AT_iChannel1_stat, _xy_), AT_iChannel1_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel1_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel1)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureSampleRect(_xy_, _dxy_)     AT_iChannel1_bufferSampleRect(textureToBuffer(AT_iChannel1_stat, _xy_), (float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel1_textureSampleRect(_xy_, _dxy_)     _bound_iChannel1
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureSampleRectClip(_xy_, _dxy_) AT_iChannel1_bufferSampleRectClip(textureToBuffer(AT_iChannel1_stat, _xy_), (float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel1_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel1_stat, textureToBuffer(AT_iChannel1_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel1_stat->resolution.x, AT_iChannel1_stat->resolution.y)) * AT_iChannel1_stat->buffer_to_image.lo, _bound_iChannel1)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferToImage(_xy_)        (bufferToImage(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_imageToWorld(_xy_) (imageToWorld(AT_iChannel1_stat, _xy_))
#else
#define AT_iChannel1_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel1_stat, _xyz_))
#else
#define AT_iChannel1_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldToImage(_xyz_)        (worldToImage(AT_iChannel1_stat, _xyz_))
#else
#define AT_iChannel1_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel1
#define AT_iChannel1_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel1_stat, _xyz_))
#else
#define AT_iChannel1_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_data       _bound_iChannel2
#else
#define AT_iChannel2_data       0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bound      1
#else
#define AT_iChannel2_bound      0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_stat       ((global IMX_Stat * restrict) _bound_iChannel2_stat_void)
#else
#define AT_iChannel2_stat       0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_layer      &_bound_iChannel2_layer
#else
#define AT_iChannel2_layer      0
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_border     _bound_iChannel2_border
#else
#define AT_iChannel2_border     IMX_WRAP
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_storage    _bound_iChannel2_storage
#else
#define AT_iChannel2_storage    FLOAT32
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_channels   _bound_iChannel2_channels
#else
#define AT_iChannel2_channels   4
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_tuplesize  _bound_iChannel2_channels
#else
#define AT_iChannel2_tuplesize  4
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_xres       _bound_iChannel2_layer.stat->resolution.x
#else
#define AT_iChannel2_xres       1
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_yres       _bound_iChannel2_layer.stat->resolution.y
#else
#define AT_iChannel2_yres       1
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_res        convert_float2(_bound_iChannel2_layer.stat->resolution)
#else
#define AT_iChannel2_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel2
#define iChannel2_args2 CONSTANT_(_bound_iChannel2_storage), _bound_iChannel2_channels
#else
#define iChannel2_args2 _bound_iChannel2_storage, _bound_iChannel2_channels
#endif
#define iChannel2_args3 _bound_iChannel2_border, iChannel2_args2
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3)
#else
#define AT_iChannel2_bufferIndex(_xy_)  _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3)
#else
#define AT_iChannel2_bufferSample(_xy_) _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel2_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel2_stat, _xy_) + 0.5f), iChannel2_args3)
#else
#define AT_iChannel2_imageNearest(_xy_) _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, _xy_), iChannel2_args3)
#else
#define AT_iChannel2_imageSample(_xy_)  _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel2_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel2_stat, worldToImage(AT_iChannel2_stat, _xyz_)) + 0.5f), iChannel2_args3)
#else
#define AT_iChannel2_worldNearest(_xyz_)        _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, worldToImage(AT_iChannel2_stat, _xyz_)), iChannel2_args3)
#else
#define AT_iChannel2_worldSample(_xyz_) _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel2_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel2_stat, _xy_) + 0.5f), iChannel2_args3)
#else
#define AT_iChannel2_textureNearest(_xy_)       _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel2_layer, textureToBuffer(AT_iChannel2_stat, _xy_), iChannel2_args3)
#else
#define AT_iChannel2_textureSample(_xy_)        _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_1(_xy_)    bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, _xy_), iChannel2_args3)
#else
#define AT_iChannel2_1(_xy_)    _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#ifdef ALIGNED_iChannel2
#define AT_iChannel2    _bufferIndexLinF4(&_bound_iChannel2_layer, _bound_idx, iChannel2_args2)
#else
#define AT_iChannel2    bufferSampleF4(&_bound_iChannel2_layer, imageToBuffer(AT_iChannel2_stat, _bound_P_image), iChannel2_args3)
#endif
#else
#define AT_iChannel2    _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#ifdef ALIGNED_iChannel2
#define AT_iChannel2_dCdx       dCdxF4aligned(&_bound_iChannel2_layer, (int2)(_bound_gidx, _bound_gidy), iChannel2_args3)
#else
#define AT_iChannel2_dCdx       dCdxF4(&_bound_iChannel2_layer, _bound_P_image, iChannel2_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel2_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel2_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel2
#ifdef ALIGNED_iChannel2
#define AT_iChannel2_dCdy       dCdyF4aligned(&_bound_iChannel2_layer, (int2)(_bound_gidx, _bound_gidy), iChannel2_args3)
#else
#define AT_iChannel2_dCdy       dCdyF4(&_bound_iChannel2_layer, _bound_P_image, iChannel2_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel2_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel2_layer, _xy_, iChannel2_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel2_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel2_layer, _xy_, _dxy_, iChannel2_args3)
#else
#define AT_iChannel2_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel2_layer, _xy_, _dxy_, iChannel2_args2)
#else
#define AT_iChannel2_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel2_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y)), _bound_iChannel2)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageSampleRect(_xy_, _dxy_)       AT_iChannel2_bufferSampleRect(imageToBuffer(AT_iChannel2_stat, _xy_), AT_iChannel2_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel2_imageSampleRect(_xy_, _dxy_)       _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel2_bufferSampleRectClip(imageToBuffer(AT_iChannel2_stat, _xy_), AT_iChannel2_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel2_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel2)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureSampleRect(_xy_, _dxy_)     AT_iChannel2_bufferSampleRect(textureToBuffer(AT_iChannel2_stat, _xy_), (float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel2_textureSampleRect(_xy_, _dxy_)     _bound_iChannel2
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureSampleRectClip(_xy_, _dxy_) AT_iChannel2_bufferSampleRectClip(textureToBuffer(AT_iChannel2_stat, _xy_), (float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel2_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel2_stat, textureToBuffer(AT_iChannel2_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel2_stat->resolution.x, AT_iChannel2_stat->resolution.y)) * AT_iChannel2_stat->buffer_to_image.lo, _bound_iChannel2)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferToImage(_xy_)        (bufferToImage(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_imageToWorld(_xy_) (imageToWorld(AT_iChannel2_stat, _xy_))
#else
#define AT_iChannel2_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel2_stat, _xyz_))
#else
#define AT_iChannel2_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldToImage(_xyz_)        (worldToImage(AT_iChannel2_stat, _xyz_))
#else
#define AT_iChannel2_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel2
#define AT_iChannel2_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel2_stat, _xyz_))
#else
#define AT_iChannel2_worldToImage3(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_data       _bound_iChannel3
#else
#define AT_iChannel3_data       0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bound      1
#else
#define AT_iChannel3_bound      0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_stat       ((global IMX_Stat * restrict) _bound_iChannel3_stat_void)
#else
#define AT_iChannel3_stat       0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_layer      &_bound_iChannel3_layer
#else
#define AT_iChannel3_layer      0
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_border     _bound_iChannel3_border
#else
#define AT_iChannel3_border     IMX_WRAP
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_storage    _bound_iChannel3_storage
#else
#define AT_iChannel3_storage    FLOAT32
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_channels   _bound_iChannel3_channels
#else
#define AT_iChannel3_channels   4
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_tuplesize  _bound_iChannel3_channels
#else
#define AT_iChannel3_tuplesize  4
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_xres       _bound_iChannel3_layer.stat->resolution.x
#else
#define AT_iChannel3_xres       1
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_yres       _bound_iChannel3_layer.stat->resolution.y
#else
#define AT_iChannel3_yres       1
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_res        convert_float2(_bound_iChannel3_layer.stat->resolution)
#else
#define AT_iChannel3_res        (float2)(1)
#endif
#ifdef CONSTANT_iChannel3
#define iChannel3_args2 CONSTANT_(_bound_iChannel3_storage), _bound_iChannel3_channels
#else
#define iChannel3_args2 _bound_iChannel3_storage, _bound_iChannel3_channels
#endif
#define iChannel3_args3 _bound_iChannel3_border, iChannel3_args2
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferIndex(_xy_)  bufferIndexF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3)
#else
#define AT_iChannel3_bufferIndex(_xy_)  _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferSample(_xy_) bufferSampleF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3)
#else
#define AT_iChannel3_bufferSample(_xy_) _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageNearest(_xy_) bufferIndexF4(&_bound_iChannel3_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel3_stat, _xy_) + 0.5f), iChannel3_args3)
#else
#define AT_iChannel3_imageNearest(_xy_) _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageSample(_xy_)  bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, _xy_), iChannel3_args3)
#else
#define AT_iChannel3_imageSample(_xy_)  _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldNearest(_xyz_)        bufferIndexF4(&_bound_iChannel3_layer, convert_int2_sat_rtn(imageToBuffer(AT_iChannel3_stat, worldToImage(AT_iChannel3_stat, _xyz_)) + 0.5f), iChannel3_args3)
#else
#define AT_iChannel3_worldNearest(_xyz_)        _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldSample(_xyz_) bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, worldToImage(AT_iChannel3_stat, _xyz_)), iChannel3_args3)
#else
#define AT_iChannel3_worldSample(_xyz_) _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureNearest(_xy_)       bufferIndexF4(&_bound_iChannel3_layer, convert_int2_sat_rtn(textureToBuffer(AT_iChannel3_stat, _xy_) + 0.5f), iChannel3_args3)
#else
#define AT_iChannel3_textureNearest(_xy_)       _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureSample(_xy_)        bufferSampleF4(&_bound_iChannel3_layer, textureToBuffer(AT_iChannel3_stat, _xy_), iChannel3_args3)
#else
#define AT_iChannel3_textureSample(_xy_)        _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_1(_xy_)    bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, _xy_), iChannel3_args3)
#else
#define AT_iChannel3_1(_xy_)    _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#ifdef ALIGNED_iChannel3
#define AT_iChannel3    _bufferIndexLinF4(&_bound_iChannel3_layer, _bound_idx, iChannel3_args2)
#else
#define AT_iChannel3    bufferSampleF4(&_bound_iChannel3_layer, imageToBuffer(AT_iChannel3_stat, _bound_P_image), iChannel3_args3)
#endif
#else
#define AT_iChannel3    _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#ifdef ALIGNED_iChannel3
#define AT_iChannel3_dCdx       dCdxF4aligned(&_bound_iChannel3_layer, (int2)(_bound_gidx, _bound_gidy), iChannel3_args3)
#else
#define AT_iChannel3_dCdx       dCdxF4(&_bound_iChannel3_layer, _bound_P_image, iChannel3_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel3_dCdx       ((float4)0)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_dCdx_1(_xy_)       dCdxF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel3_dCdx_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel3
#ifdef ALIGNED_iChannel3
#define AT_iChannel3_dCdy       dCdyF4aligned(&_bound_iChannel3_layer, (int2)(_bound_gidx, _bound_gidy), iChannel3_args3)
#else
#define AT_iChannel3_dCdy       dCdyF4(&_bound_iChannel3_layer, _bound_P_image, iChannel3_args3, &_RUNOVER_LAYER)
#endif
#else
#define AT_iChannel3_dCdy       ((float4)0)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_dCdy_1(_xy_)       dCdyF4(&_bound_iChannel3_layer, _xy_, iChannel3_args3, &_RUNOVER_LAYER)
#else
#define AT_iChannel3_dCdy_1(_xy_)       ((float4)0)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferSampleRect(_xy_, _dxy_)      bufferSampleRectF4(&_bound_iChannel3_layer, _xy_, _dxy_, iChannel3_args3)
#else
#define AT_iChannel3_bufferSampleRect(_xy_, _dxy_)      _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferSampleRectClip(_xy_, _dxy_)  bufferSampleRectClipF4(&_bound_iChannel3_layer, _xy_, _dxy_, iChannel3_args2)
#else
#define AT_iChannel3_bufferSampleRectClip(_xy_, _dxy_)  constImageSampleRectClip(bufferToImage(AT_iChannel3_stat, _xy_), _dxy_ * (0.5f / (float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y)), _bound_iChannel3)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageSampleRect(_xy_, _dxy_)       AT_iChannel3_bufferSampleRect(imageToBuffer(AT_iChannel3_stat, _xy_), AT_iChannel3_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel3_imageSampleRect(_xy_, _dxy_)       _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageSampleRectClip(_xy_, _dxy_)   AT_iChannel3_bufferSampleRectClip(imageToBuffer(AT_iChannel3_stat, _xy_), AT_iChannel3_stat->image_to_buffer.lo * (_dxy_))
#else
#define AT_iChannel3_imageSampleRectClip(_xy_, _dxy_)   constImageSampleRectClip(_xy_, _dxy_, _bound_iChannel3)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureSampleRect(_xy_, _dxy_)     AT_iChannel3_bufferSampleRect(textureToBuffer(AT_iChannel3_stat, _xy_), (float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel3_textureSampleRect(_xy_, _dxy_)     _bound_iChannel3
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureSampleRectClip(_xy_, _dxy_) AT_iChannel3_bufferSampleRectClip(textureToBuffer(AT_iChannel3_stat, _xy_), (float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y) * (_dxy_))
#else
#define AT_iChannel3_textureSampleRectClip(_xy_, _dxy_) constImageSampleRectClip(bufferToImage(AT_iChannel3_stat, textureToBuffer(AT_iChannel3_stat, _xy_)), _dxy_ * ((float2)(AT_iChannel3_stat->resolution.x, AT_iChannel3_stat->resolution.y)) * AT_iChannel3_stat->buffer_to_image.lo, _bound_iChannel3)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferToImage(_xy_)        (bufferToImage(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_bufferToImage(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageToBuffer(_xy_)        (imageToBuffer(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_imageToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferToPixel(_xy_)        (bufferToPixel(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_bufferToPixel(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_pixelToBuffer(_xy_)        (pixelToBuffer(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_pixelToBuffer(_xy_)        (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_bufferToTexture(_xy_)      (bufferToTexture(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_bufferToTexture(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_textureToBuffer(_xy_)      (textureToBuffer(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_textureToBuffer(_xy_)      (_xy_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_imageToWorld(_xy_) (imageToWorld(AT_iChannel3_stat, _xy_))
#else
#define AT_iChannel3_imageToWorld(_xy_) ((float3)((_xy_).x, (_xy_).y, 0))
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_image3ToWorld(_xyz_)       (image3ToWorld(AT_iChannel3_stat, _xyz_))
#else
#define AT_iChannel3_image3ToWorld(_xyz_)       (_xyz_)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldToImage(_xyz_)        (worldToImage(AT_iChannel3_stat, _xyz_))
#else
#define AT_iChannel3_worldToImage(_xyz_)        ((_xyz_).xy)
#endif
#ifdef HAS_iChannel3
#define AT_iChannel3_worldToImage3(_xyz_)       (worldToImage3(AT_iChannel3_stat, _xyz_))
#else
#define AT_iChannel3_worldToImage3(_xyz_)       (_xyz_)
#endif
#define AT_fragColor_data       _bound_fragColor
#define AT_fragColor_bound      1
#define AT_fragColor_stat       ((global IMX_Stat * restrict) _bound_fragColor_stat_void)
#define AT_fragColor_layer      &_bound_fragColor_layer
#define AT_fragColor_border     _bound_fragColor_border
#define AT_fragColor_storage    _bound_fragColor_storage
#define AT_fragColor_channels   _bound_fragColor_channels
#define AT_fragColor_tuplesize  _bound_fragColor_channels
#define AT_fragColor_xres       _bound_fragColor_layer.stat->resolution.x
#define AT_fragColor_yres       _bound_fragColor_layer.stat->resolution.y
#define AT_fragColor_res        convert_float2(_bound_fragColor_layer.stat->resolution)
#define AT_fragColor_set(_val_) _setIndexLinF4(&_bound_fragColor_layer, _bound_idx, _val_, _bound_fragColor_storage, _bound_fragColor_channels)
#define AT_fragColor_setIndex(_xy_, _val_)      _setIndexF4(&_bound_fragColor_layer, _xy_, _val_, _bound_fragColor_storage, _bound_fragColor_channels)
#define AT_fragColor_bufferToImage(_xy_)        (bufferToImage(AT_fragColor_stat, _xy_))
#define AT_fragColor_imageToBuffer(_xy_)        (imageToBuffer(AT_fragColor_stat, _xy_))
#define AT_fragColor_bufferToPixel(_xy_)        (bufferToPixel(AT_fragColor_stat, _xy_))
#define AT_fragColor_pixelToBuffer(_xy_)        (pixelToBuffer(AT_fragColor_stat, _xy_))
#define AT_fragColor_bufferToTexture(_xy_)      (bufferToTexture(AT_fragColor_stat, _xy_))
#define AT_fragColor_textureToBuffer(_xy_)      (textureToBuffer(AT_fragColor_stat, _xy_))
#define AT_fragColor_imageToWorld(_xy_) (imageToWorld(AT_fragColor_stat, _xy_))
#define AT_fragColor_image3ToWorld(_xyz_)       (image3ToWorld(AT_fragColor_stat, _xyz_))
#define AT_fragColor_worldToImage(_xyz_)        (worldToImage(AT_fragColor_stat, _xyz_))
#define AT_fragColor_worldToImage3(_xyz_)       (worldToImage3(AT_fragColor_stat, _xyz_))
#line 1

// ---- Simplified GLSL Helper Functions ----
// Simple, reliable helper functions for common operations
#include "glslHelpers.h"

// ---- Shadertoy-like texture() for Copernicus ----
#include "textureHelpers.h"

// Shadertoy has global variables that can be called inside functions
// We just initiate empty variables so that code compiles if used inside func()
// They get mapped inside kernel
static float3 iResolution = (float3)(512.0f, 288.0f, 0.0f);
static float iTime = 0.0000f;
static float iTimeDelta = 0.0000f;
static float iFrameRate = 24.0000f;
static int iFrame = 0;
static float4 iMouse = (float4)(0.0000f, 0.0000f, 0.0000f, 0.0000f );
static float4 iDate = (float4)(2025.0000f, 12.0000f, 31.0000f, 60.0000f );
static const float iSampleRate = 44100.0f;

static const IMX_Layer* iChannel0;
static const IMX_Layer* iChannel1;
static const IMX_Layer* iChannel2;
static const IMX_Layer* iChannel3;
static float iChannelTime[4];
static float3 iChannelResolution[4];


#ifdef CUBEMAP_RENDERPASS
    #define DO_CUBEMAP \
        float3 rayDir; \
        shadertoy_cubemap(AT_ix,AT_iy,AT_xres,AT_yres,&rayDir,&iResolution);
#else
    #define DO_CUBEMAP /* nothing */
#endif

#define SHADERTOY_INPUTS \
    iResolution = (float3)(AT_xres, AT_yres, 0.0f); \
    iTime = AT_Time; \
    iFrameRate = AT_iFrameRate; \
    iFrame = AT_iFrame; \
    iMouse = AT_iMouse;\
    iDate = AT_iDate;\
    iChannel0 = AT_iChannel0_layer; \
    iChannel1 = AT_iChannel1_layer; \
    iChannel2 = AT_iChannel2_layer; \
    iChannel3 = AT_iChannel3_layer; \
    iChannelTime[0] = AT_Time; \
    iChannelTime[1] = AT_Time; \
    iChannelTime[2] = AT_Time; \
    iChannelTime[3] = AT_Time; \
    iChannelResolution[0] = (float3)(AT_iChannel0_res, 0.0f); \
    iChannelResolution[1] = (float3)(AT_iChannel1_res, 0.0f); \
    iChannelResolution[2] = (float3)(AT_iChannel2_res, 0.0f); \
    iChannelResolution[3] = (float3)(AT_iChannel3_res, 0.0f); \
    float2 fragCoord = AT_fragCoord; \
    if (!AT_fragCoord_bound) { fragCoord = (float2)(AT_ix, AT_iy); }\
    float4 fragColor = (float4)(0.0f, 0.0f, 0.0f, 1.0f); \
    DO_CUBEMAP

// mainCubemap renderpass helper    
// Unpacks 3x2 cubemap layout to ray direction and adjusts resolution to cube face
// Standard cubemap layout:
//   [+X][-X][+Z]
//   [+Y][-Y][-Z]
static void shadertoy_cubemap(int ix, int iy, int xres, int yres, 
                              float3* rayDir, float3* iResolution)
{
    // Calculate individual face dimensions
    int face_width = xres / 3;
    int face_height = yres / 2;
    
    // Update iResolution to single face size
    *iResolution = (float3)(face_width, face_height, 0.0f);
    
    // Determine which face we're rendering (0-2 for x, 0-1 for y)
    int face_x = ix / face_width;
    int face_y = iy / face_height;
    
    // Calculate local UV coordinates within the face (-1 to 1 range)
    float2 local_uv = (float2)(
        (float)(ix % face_width) / (float)face_width * 2.0f - 1.0f,
        (float)(iy % face_height) / (float)face_height * 2.0f - 1.0f
    );
    
    // Map face position to ray direction
    // Each face represents a direction in the cube
    if (face_x == 0 && face_y == 0) {
        // +X face (right)
        *rayDir = (float3)(1.0f, -local_uv.y, -local_uv.x);
    } 
    else if (face_x == 1 && face_y == 0) {
        // -X face (left)
        *rayDir = (float3)(-1.0f, -local_uv.y, local_uv.x);
    } 
    else if (face_x == 0 && face_y == 1) {
        // +Y face (up)
        *rayDir = (float3)(local_uv.x, 1.0f, local_uv.y);
    } 
    else if (face_x == 1 && face_y == 1) {
        // -Y face (down)
        *rayDir = (float3)(local_uv.x, -1.0f, -local_uv.y);
    }
    else if (face_x == 2 && face_y == 0) {
        // +Z face (forward)
        *rayDir = (float3)(local_uv.x, -local_uv.y, 1.0f);
    } 
    else if (face_x == 2 && face_y == 1) {
        // -Z face (back)
        *rayDir = (float3)(-local_uv.x, -local_uv.y, -1.0f);
    }
}





