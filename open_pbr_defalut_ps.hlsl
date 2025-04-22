// HLSL Pixel Shader for PBR rendering
// Translated from GLSL to HLSL

// Constants
#define M_FLOAT_EPS 1e-8
#define M_PI 3.1415926535897932
#define M_PI_INV (1.0 / M_PI)
#define MAX_LIGHT_SOURCES 3
#define DIRECTIONAL_ALBEDO_METHOD 0

// Struct definitions
struct BSDF {
    float3 response;
    float3 throughput;
};

#define EDF float3

struct SurfaceShader {
    float3 color;
    float3 transparency;
};

struct volumeshader {
    float3 color;
    float3 transparency;
};

struct displacementshader {
    float3 offset;
    float scale;
};

struct LightShader {
    float3 intensity;
    float3 direction;
};

#define material SurfaceShader

// Constant buffers (replacing GLSL uniforms)
cbuffer PrivateUniforms : register(b0) {
    float4x4 u_shadowMatrix;
    float4x4 u_envMatrix;
    float u_envLightIntensity;
    int u_envRadianceMips;
    int u_envRadianceSamples;
    bool u_refractionTwoSided;
    float3 u_viewPosition;
    int u_numActiveLightSources;
};

cbuffer PublicUniforms : register(b1) {
    SurfaceShader backsurfaceshader;
    displacementshader displacementshader1;
    float open_pbr_surface_surfaceshader_base_weight;
    float3 open_pbr_surface_surfaceshader_base_color;
    float open_pbr_surface_surfaceshader_base_diffuse_roughness;
    float open_pbr_surface_surfaceshader_base_metalness;
    float open_pbr_surface_surfaceshader_specular_weight;
    float3 open_pbr_surface_surfaceshader_specular_color;
    float open_pbr_surface_surfaceshader_specular_roughness;
    float open_pbr_surface_surfaceshader_specular_ior;
    float open_pbr_surface_surfaceshader_specular_roughness_anisotropy;
    float open_pbr_surface_surfaceshader_transmission_weight;
    float3 open_pbr_surface_surfaceshader_transmission_color;
    float open_pbr_surface_surfaceshader_transmission_depth;
    float3 open_pbr_surface_surfaceshader_transmission_scatter;
    float open_pbr_surface_surfaceshader_transmission_scatter_anisotropy;
    float open_pbr_surface_surfaceshader_transmission_dispersion_scale;
    float open_pbr_surface_surfaceshader_transmission_dispersion_abbe_number;
    float open_pbr_surface_surfaceshader_subsurface_weight;
    float3 open_pbr_surface_surfaceshader_subsurface_color;
    float open_pbr_surface_surfaceshader_subsurface_radius;
    float3 open_pbr_surface_surfaceshader_subsurface_radius_scale;
    float open_pbr_surface_surfaceshader_subsurface_scatter_anisotropy;
    float open_pbr_surface_surfaceshader_fuzz_weight;
    float3 open_pbr_surface_surfaceshader_fuzz_color;
    float open_pbr_surface_surfaceshader_fuzz_roughness;
    float open_pbr_surface_surfaceshader_coat_weight;
    float3 open_pbr_surface_surfaceshader_coat_color;
    float open_pbr_surface_surfaceshader_coat_roughness;
    float open_pbr_surface_surfaceshader_coat_roughness_anisotropy;
    float open_pbr_surface_surfaceshader_coat_ior;
    float open_pbr_surface_surfaceshader_coat_darkening;
    float open_pbr_surface_surfaceshader_thin_film_weight;
    float open_pbr_surface_surfaceshader_thin_film_thickness;
    float open_pbr_surface_surfaceshader_thin_film_ior;
    float open_pbr_surface_surfaceshader_emission_luminance;
    float3 open_pbr_surface_surfaceshader_emission_color;
    float open_pbr_surface_surfaceshader_geometry_opacity;
    bool open_pbr_surface_surfaceshader_geometry_thin_walled;
};

// Textures and Samplers
Texture2D u_shadowMap : register(t0);
Texture2D u_envRadiance : register(t1);
Texture2D u_envIrradiance : register(t2);
SamplerState u_sampler : register(s0);
SamplerState u_shadowSampler : register(s1);

// Input struct (replacing GLSL VertexData)
struct PSInput {
    float4 position : SV_POSITION;
    float3 normalWorld : NORMAL;
    float3 tangentWorld : TANGENT;
    float3 positionWorld : POSITION;
};

// Helper functions
#define mx_mod fmod
#define mx_inverse transpose // Note: HLSL uses transpose for matrix inverse in some contexts
#define mx_inversesqrt rsqrt
#define mx_sin sin
#define mx_cos cos
#define mx_tan tan
#define mx_asin asin
#define mx_acos acos
#define mx_atan atan2
#define mx_radians radians

float mx_square(float x) {
    return x * x;
}

float2 mx_square(float2 x) {
    return x * x;
}

float3 mx_square(float3 x) {
    return x * x;
}

float3 mx_srgb_encode(float3 color) {
    bool3 isAbove = color > 0.0031308;
    float3 linSeg = color * 12.92;
    float3 powSeg = 1.055 * pow(max(color, 0.0), 1.0 / 2.4) - 0.055;
    return lerp(linSeg, powSeg, isAbove);
}

float mx_pow5(float x) {
    return mx_square(mx_square(x)) * x;
}

float mx_pow6(float x) {
    float x2 = mx_square(x);
    return mx_square(x2) * x2;
}

// Standard Schlick Fresnel
float mx_fresnel_schlick(float cosTheta, float F0) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return F0 + (1.0 - F0) * x5;
}

float3 mx_fresnel_schlick(float cosTheta, float3 F0) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return F0 + (1.0 - F0) * x5;
}

// Generalized Schlick Fresnel
float mx_fresnel_schlick(float cosTheta, float F0, float F90) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return lerp(F0, F90, x5);
}

float3 mx_fresnel_schlick(float cosTheta, float3 F0, float3 F90) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x5 = mx_pow5(x);
    return lerp(F0, F90, x5);
}

// Generalized Schlick Fresnel with a variable exponent
float mx_fresnel_schlick(float cosTheta, float F0, float F90, float exponent) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    return lerp(F0, F90, pow(x, exponent));
}

float3 mx_fresnel_schlick(float cosTheta, float3 F0, float3 F90, float exponent) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    return lerp(F0, F90, pow(x, exponent));
}

// Enforce forward-facing normal
float3 mx_forward_facing_normal(float3 N, float3 V) {
    return dot(N, V) < 0.0 ? -N : N;
}

// Golden ratio sequence
float mx_golden_ratio_sequence(int i) {
    const float GOLDEN_RATIO = 1.6180339887498948;
    return frac((float(i) + 1.0) * GOLDEN_RATIO);
}

// Spherical Fibonacci
float2 mx_spherical_fibonacci(int i, int numSamples) {
    return float2((float(i) + 0.5) / float(numSamples), mx_golden_ratio_sequence(i));
}

// Uniform hemisphere sampling
float3 mx_uniform_sample_hemisphere(float2 Xi) {
    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = 1.0 - Xi.y;
    float sinTheta = sqrt(1.0 - mx_square(cosTheta));
    return float3(mx_cos(phi) * sinTheta, mx_sin(phi) * sinTheta, cosTheta);
}

// Cosine-weighted hemisphere sampling
float3 mx_cosine_sample_hemisphere(float2 Xi) {
    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = sqrt(Xi.y);
    float sinTheta = sqrt(1.0 - Xi.y);
    return float3(mx_cos(phi) * sinTheta, mx_sin(phi) * sinTheta, cosTheta);
}

// Orthonormal basis
float3x3 mx_orthonormal_basis(float3 N) {
    float s = N.z < 0.0 ? -1.0 : 1.0;
    float a = -1.0 / (s + N.z);
    float b = N.x * N.y * a;
    float3 X = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x);
    float3 Y = float3(b, s + N.y * N.y * a, -N.y);
    return float3x3(X, Y, N);
}

// Fresnel parameters
#define FRESNEL_MODEL_DIELECTRIC 0
#define FRESNEL_MODEL_CONDUCTOR 1
#define FRESNEL_MODEL_SCHLICK 2

struct FresnelData {
    int model;
    bool airy;
    float3 ior;
    float3 extinction;
    float3 F0;
    float3 F82;
    float3 F90;
    float exponent;
    float tf_thickness;
    float tf_ior;
    bool refraction;
};

// GGX NDF
float mx_ggx_NDF(float3 H, float2 alpha) {
    float2 He = H.xy / alpha;
    float denom = dot(He, He) + mx_square(H.z);
    return 1.0 / (M_PI * alpha.x * alpha.y * mx_square(denom));
}

// GGX VNDF importance sampling
float3 mx_ggx_importance_sample_VNDF(float2 Xi, float3 V, float2 alpha) {
    V = normalize(float3(V.xy * alpha, V.z));
    float phi = 2.0 * M_PI * Xi.x;
    float z = (1.0 - Xi.y) * (1.0 + V.z) - V.z;
    float sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    float x = sinTheta * mx_cos(phi);
    float y = sinTheta * mx_sin(phi);
    float3 c = float3(x, y, z);
    float3 H = c + V;
    H = normalize(float3(H.xy * alpha, max(H.z, 0.0)));
    return H;
}

// Smith G1
float mx_ggx_smith_G1(float cosTheta, float alpha) {
    float cosTheta2 = mx_square(cosTheta);
    float tanTheta2 = (1.0 - cosTheta2) / cosTheta2;
    return 2.0 / (1.0 + sqrt(1.0 + mx_square(alpha) * tanTheta2));
}

// Smith G2
float mx_ggx_smith_G2(float NdotL, float NdotV, float alpha) {
    float alpha2 = mx_square(alpha);
    float lambdaL = sqrt(alpha2 + (1.0 - alpha2) * mx_square(NdotL));
    float lambdaV = sqrt(alpha2 + (1.0 - alpha2) * mx_square(NdotV));
    return 2.0 / (lambdaL / NdotL + lambdaV / NdotV);
}

// GGX directional albedo (analytic)
float3 mx_ggx_dir_albedo_analytic(float NdotV, float alpha, float3 F0, float3 F90) {
    float x = NdotV;
    float y = alpha;
    float x2 = mx_square(x);
    float y2 = mx_square(y);
    float4 r = float4(0.1003, 0.9345, 1.0, 1.0) +
               float4(-0.6303, -2.323, -1.765, 0.2281) * x +
               float4(9.748, 2.229, 8.263, 15.94) * y +
               float4(-2.038, -3.748, 11.53, -55.83) * x * y +
               float4(29.34, 1.424, 28.96, 13.08) * x2 +
               float4(-8.245, -0.7684, -7.507, 41.26) * y2 +
               float4(-26.44, 1.436, -36.11, 54.9) * x2 * y +
               float4(19.99, 0.2913, 15.86, 300.2) * x * y2 +
               float4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
    float2 AB = clamp(r.xy / r.zw, 0.0, 1.0);
    return F0 * AB.x + F90 * AB.y;
}

// GGX directional albedo (Monte Carlo)
float3 mx_ggx_dir_albedo_monte_carlo(float NdotV, float alpha, float3 F0, float3 F90) {
    NdotV = clamp(NdotV, M_FLOAT_EPS, 1.0);
    float3 V = float3(sqrt(1.0 - mx_square(NdotV)), 0, NdotV);
    float2 AB = float2(0.0, 0.0);
    const int SAMPLE_COUNT = 64;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float2 Xi = mx_spherical_fibonacci(i, SAMPLE_COUNT);
        float3 H = mx_ggx_importance_sample_VNDF(Xi, V, float2(alpha, alpha));
        float3 L = -reflect(V, H);
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);
        float Fc = mx_fresnel_schlick(VdotH, 0.0, 1.0);
        float G2 = mx_ggx_smith_G2(NdotL, NdotV, alpha);
        AB += float2(G2 * (1.0 - Fc), G2 * Fc);
    }
    AB /= mx_ggx_smith_G1(NdotV, alpha) * float(SAMPLE_COUNT);
    return F0 * AB.x + F90 * AB.y;
}

float3 mx_ggx_dir_albedo(float NdotV, float alpha, float3 F0, float3 F90) {
    #if DIRECTIONAL_ALBEDO_METHOD == 0
        return mx_ggx_dir_albedo_analytic(NdotV, alpha, F0, F90);
    #else
        return mx_ggx_dir_albedo_monte_carlo(NdotV, alpha, F0, F90);
    #endif
}

float mx_ggx_dir_albedo(float NdotV, float alpha, float F0, float F90) {
    return mx_ggx_dir_albedo(NdotV, alpha, float3(F0, F0, F0), float3(F90, F90, F90)).x;
}

// GGX energy compensation
float3 mx_ggx_energy_compensation(float NdotV, float alpha, float3 Fss) {
    float Ess = mx_ggx_dir_albedo(NdotV, alpha, 1.0, 1.0);
    return 1.0 + Fss * (1.0 - Ess) / Ess;
}

float mx_ggx_energy_compensation(float NdotV, float alpha, float Fss) {
    return mx_ggx_energy_compensation(NdotV, alpha, float3(Fss, Fss, Fss)).x;
}

// Average alpha
float mx_average_alpha(float2 alpha) {
    return sqrt(alpha.x * alpha.y);
}

// IOR to F0
float mx_ior_to_f0(float ior) {
    return mx_square((ior - 1.0) / (ior + 1.0));
}

float mx_f0_to_ior(float F0) {
    float sqrtF0 = sqrt(clamp(F0, 0.01, 0.99));
    return (1.0 + sqrtF0) / (1.0 - sqrtF0);
}

float3 mx_f0_to_ior(float3 F0) {
    float3 sqrtF0 = sqrt(clamp(F0, 0.01, 0.99));
    return (1.0 + sqrtF0) / (1.0 - sqrtF0);
}

// Hoffman-Schlick Fresnel
float3 mx_fresnel_hoffman_schlick(float cosTheta, FresnelData fd) {
    const float COS_THETA_MAX = 1.0 / 7.0;
    const float COS_THETA_FACTOR = 1.0 / (COS_THETA_MAX * pow(1.0 - COS_THETA_MAX, 6.0));
    float x = clamp(cosTheta, 0.0, 1.0);
    float3 a = lerp(fd.F0, fd.F90, pow(1.0 - COS_THETA_MAX, fd.exponent)) * (1.0 - fd.F82) * COS_THETA_FACTOR;
    return lerp(fd.F0, fd.F90, pow(1.0 - x, fd.exponent)) - a * x * mx_pow6(1.0 - x);
}

// Dielectric Fresnel
float mx_fresnel_dielectric(float cosTheta, float ior) {
    float c = cosTheta;
    float g2 = ior * ior + c * c - 1.0;
    if (g2 < 0.0) {
        return 1.0;
    }
    float g = sqrt(g2);
    return 0.5 * mx_square((g - c) / (g + c)) * (1.0 + mx_square(((g + c) * c - 1.0) / ((g - c) * c + 1.0)));
}

// Dielectric polarized Fresnel
float2 mx_fresnel_dielectric_polarized(float cosTheta, float ior) {
    float cosTheta2 = mx_square(clamp(cosTheta, 0.0, 1.0));
    float sinTheta2 = 1.0 - cosTheta2;
    float t0 = max(ior * ior - sinTheta2, 0.0);
    float t1 = t0 + cosTheta2;
    float t2 = 2.0 * sqrt(t0) * cosTheta;
    float Rs = (t1 - t2) / (t1 + t2);
    float t3 = cosTheta2 * t0 + sinTheta2 * sinTheta2;
    float t4 = t2 * sinTheta2;
    float Rp = Rs * (t3 - t4) / (t3 + t4);
    return float2(Rp, Rs);
}

// Conductor polarized Fresnel
void mx_fresnel_conductor_polarized(float cosTheta, float3 n, float3 k, out float3 Rp, out float3 Rs) {
    float cosTheta2 = mx_square(clamp(cosTheta, 0.0, 1.0));
    float sinTheta2 = 1.0 - cosTheta2;
    float3 n2 = n * n;
    float3 k2 = k * k;
    float3 t0 = n2 - k2 - sinTheta2;
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
    float3 t1 = a2plusb2 + cosTheta2;
    float3 a = sqrt(max(0.5 * (a2plusb2 + t0), 0.0));
    float3 t2 = 2.0 * a * cosTheta;
    Rs = (t1 - t2) / (t1 + t2);
    float3 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    float3 t4 = t2 * sinTheta2;
    Rp = Rs * (t3 - t4) / (t3 + t4);
}

float3 mx_fresnel_conductor(float cosTheta, float3 n, float3 k) {
    float3 Rp, Rs;
    mx_fresnel_conductor_polarized(cosTheta, n, k, Rp, Rs);
    return 0.5 * (Rp + Rs);
}

// Conductor phase polarized
void mx_fresnel_conductor_phase_polarized(float cosTheta, float eta1, float3 eta2, float3 kappa2, out float3 phiP, out float3 phiS) {
    float3 k2 = kappa2 / eta2;
    float3 sinThetaSqr = 1.0 - cosTheta * cosTheta;
    float3 A = eta2 * eta2 * (1.0 - k2 * k2) - eta1 * eta1 * sinThetaSqr;
    float3 B = sqrt(A * A + mx_square(2.0 * eta2 * eta2 * k2));
    float3 U = sqrt((A + B) / 2.0);
    float3 V = max(0.0, sqrt((B - A) / 2.0));
    phiS = mx_atan(2.0 * eta1 * V * cosTheta, U * U + V * V - mx_square(eta1 * cosTheta));
    phiP = mx_atan(2.0 * eta1 * eta2 * eta2 * cosTheta * (2.0 * k2 * U - (1.0 - k2 * k2) * V),
                    mx_square(eta2 * eta2 * (1.0 + k2 * k2) * cosTheta) - eta1 * eta1 * (U * U + V * V));
}

// Sensitivity evaluation
float3 mx_eval_sensitivity(float opd, float3 shift) {
    float phase = 2.0 * M_PI * opd;
    float3 val = float3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    float3 pos = float3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    float3 var = float3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    float3 xyz = val * sqrt(2.0 * M_PI * var) * mx_cos(pos * phase + shift) * exp(-var * phase * phase);
    xyz.x += 9.7470e-14 * sqrt(2.0 * M_PI * 4.5282e+09) * mx_cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * phase * phase);
    return xyz / 1.0685e-7;
}

// Airy Fresnel
float3 mx_fresnel_airy(float cosTheta, FresnelData fd) {
    const float3x3 XYZ_TO_RGB = float3x3(
        2.3706743, -0.9000405, -0.4706338,
        -0.5138850, 1.4253036, 0.0885814,
        0.0052982, -0.0146949, 1.0093968
    );
    float eta1 = 1.0;
    float eta2 = max(fd.tf_ior, eta1);
    float3 eta3 = (fd.model == FRESNEL_MODEL_SCHLICK) ? mx_f0_to_ior(fd.F0) : fd.ior;
    float3 kappa3 = (fd.model == FRESNEL_MODEL_SCHLICK) ? 0.0 : fd.extinction;
    float cosThetaT = sqrt(1.0 - (1.0 - mx_square(cosTheta)) * mx_square(eta1 / eta2));
    float2 R12 = mx_fresnel_dielectric_polarized(cosTheta, eta2 / eta1);
    if (cosThetaT <= 0.0) {
        R12 = float2(1.0, 1.0);
    }
    float2 T121 = 1.0 - R12;
    float3 R23p, R23s;
    if (fd.model == FRESNEL_MODEL_SCHLICK) {
        float3 f = mx_fresnel_hoffman_schlick(cosThetaT, fd);
        R23p = 0.5 * f;
        R23s = 0.5 * f;
    } else {
        mx_fresnel_conductor_polarized(cosThetaT, eta3 / eta2, kappa3 / eta2, R23p, R23s);
    }
    float cosB = mx_cos(atan2(eta2, eta1));
    float2 phi21 = cosTheta < cosB ? float2(0.0, M_PI) : float2(M_PI, M_PI);
    float3 phi23p, phi23s;
    if (fd.model == FRESNEL_MODEL_SCHLICK) {
        phi23p = float3((eta3[0] < eta2) ? M_PI : 0.0,
                        (eta3[1] < eta2) ? M_PI : 0.0,
                        (eta3[2] < eta2) ? M_PI : 0.0);
        phi23s = phi23p;
    } else {
        mx_fresnel_conductor_phase_polarized(cosThetaT, eta2, eta3, kappa3, phi23p, phi23s);
    }
    float3 r123p = max(sqrt(R12.x * R23p), 0.0);
    float3 r123s = max(sqrt(R12.y * R23s), 0.0);
    float3 I = 0.0;
    float3 Cm, Sm;
    float distMeters = fd.tf_thickness * 1.0e-9;
    float opd = 2.0 * eta2 * cosThetaT * distMeters;
    float3 Rs = (mx_square(T121.x) * R23p) / (1.0 - R12.x * R23p);
    I += R12.x + Rs;
    Cm = Rs - T121.x;
    for (int m = 1; m <= 2; m++) {
        Cm *= r123p;
        Sm = 2.0 * mx_eval_sensitivity(float(m) * opd, float(m) * (phi23p + phi21.x));
        I += Cm * Sm;
    }
    float3 Rp = (mx_square(T121.y) * R23s) / (1.0 - R12.y * R23s);
    I += R12.y + Rp;
    Cm = Rp - T121.y;
    for (int m = 1; m <= 2; m++) {
        Cm *= r123s;
        Sm = 2.0 * mx_eval_sensitivity(float(m) * opd, float(m) * (phi23s + phi21.y));
        I += Cm * Sm;
    }
    I *= 0.5;
    I = clamp(mul(XYZ_TO_RGB, I), 0.0, 1.0);
    return I;
}

// Fresnel initialization
FresnelData mx_init_fresnel_dielectric(float ior, float tf_thickness, float tf_ior) {
    FresnelData fd;
    fd.model = FRESNEL_MODEL_DIELECTRIC;
    fd.airy = tf_thickness > 0.0;
    fd.ior = float3(ior, ior, ior);
    fd.extinction = 0.0;
    fd.F0 = 0.0;
    fd.F82 = 0.0;
    fd.F90 = 0.0;
    fd.exponent = 0.0;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

FresnelData mx_init_fresnel_conductor(float3 ior, float3 extinction, float tf_thickness, float tf_ior) {
    FresnelData fd;
    fd.model = FRESNEL_MODEL_CONDUCTOR;
    fd.airy = tf_thickness > 0.0;
    fd.ior = ior;
    fd.extinction = extinction;
    fd.F0 = 0.0;
    fd.F82 = 0.0;
    fd.F90 = 0.0;
    fd.exponent = 0.0;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

FresnelData mx_init_fresnel_schlick(float3 F0, float3 F82, float3 F90, float exponent, float tf_thickness, float tf_ior) {
    FresnelData fd;
    fd.model = FRESNEL_MODEL_SCHLICK;
    fd.airy = tf_thickness > 0.0;
    fd.ior = 0.0;
    fd.extinction = 0.0;
    fd.F0 = F0;
    fd.F82 = F82;
    fd.F90 = F90;
    fd.exponent = exponent;
    fd.tf_thickness = tf_thickness;
    fd.tf_ior = tf_ior;
    fd.refraction = false;
    return fd;
}

float3 mx_compute_fresnel(float cosTheta, FresnelData fd) {
    if (fd.airy) {
        return mx_fresnel_airy(cosTheta, fd);
    } else if (fd.model == FRESNEL_MODEL_DIELECTRIC) {
        return float3(mx_fresnel_dielectric(cosTheta, fd.ior.x).xxx);
    } else if (fd.model == FRESNEL_MODEL_CONDUCTOR) {
        return mx_fresnel_conductor(cosTheta, fd.ior, fd.extinction);
    } else {
        return mx_fresnel_hoffman_schlick(cosTheta, fd);
    }
}

// Solid sphere refraction
float3 mx_refraction_solid_sphere(float3 R, float3 N, float ior) {
    R = refract(R, N, 1.0 / ior);
    float3 N1 = normalize(R * dot(R, N) - N * 0.5);
    return refract(R, N1, ior);
}

// Latlong projection
float2 mx_latlong_projection(float3 dir) {
    float latitude = -mx_asin(dir.y) * M_PI_INV + 0.5;
    float longitude = mx_atan(dir.x, -dir.z) * M_PI_INV * 0.5 + 0.5;
    return float2(longitude, latitude);
}

// Environment map lookup
float3 mx_latlong_map_lookup(float3 dir, float4x4 transform, float lod, Texture2D envSampler, SamplerState sampler) {
    float3 envDir = normalize(mul(transform, float4(dir, 0.0)).xyz);
    float2 uv = mx_latlong_projection(envDir);
    return envSampler.SampleLevel(sampler, uv, lod).rgb;
}

// Compute LOD for environment sampling
float mx_latlong_compute_lod(float3 dir, float pdf, float maxMipLevel, int envSamples) {
    const float MIP_LEVEL_OFFSET = 1.5;
    float effectiveMaxMipLevel = maxMipLevel - MIP_LEVEL_OFFSET;
    float distortion = sqrt(1.0 - mx_square(dir.y));
    return max(effectiveMaxMipLevel - 0.5 * log2(float(envSamples) * pdf * distortion), 0.0);
}

// Environment radiance
float3 mx_environment_radiance(float3 N, float3 V, float3 X, float2 alpha, int distribution, FresnelData fd) {
    X = normalize(X - dot(X, N) * N);
    float3 Y = cross(N, X);
    float3x3 tangentToWorld = float3x3(X, Y, N);
    V = float3(dot(V, X), dot(V, Y), dot(V, N));
    float NdotV = clamp(V.z, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(alpha);
    float G1V = mx_ggx_smith_G1(NdotV, avgAlpha);
    float3 radiance = 0.0;
    int envRadianceSamples = u_envRadianceSamples;
    for (int i = 0; i < envRadianceSamples; i++) {
        float2 Xi = mx_spherical_fibonacci(i, envRadianceSamples);
        float3 H = mx_ggx_importance_sample_VNDF(Xi, V, alpha);
        float3 L = fd.refraction ? mx_refraction_solid_sphere(-V, H, fd.ior.x) : -reflect(V, H);
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);
        float3 Lw = mul(tangentToWorld, L);
        float pdf = mx_ggx_NDF(H, alpha) * G1V / (4.0 * NdotV);
        float lod = mx_latlong_compute_lod(Lw, pdf, float(u_envRadianceMips - 1), envRadianceSamples);
        float3 sampleColor = mx_latlong_map_lookup(Lw, u_envMatrix, lod, u_envRadiance, u_sampler);
        float3 F = mx_compute_fresnel(VdotH, fd);
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);
        float3 FG = fd.refraction ? 1.0 - F : F * G;
        radiance += sampleColor * FG;
    }
    radiance /= G1V * float(envRadianceSamples);
    return radiance * u_envLightIntensity;
}

// Environment irradiance
float3 mx_environment_irradiance(float3 N) {
    float3 Li = mx_latlong_map_lookup(N, u_envMatrix, 0.0, u_envIrradiance, u_sampler);
    return Li * u_envLightIntensity;
}

// Surface transmission
float3 mx_surface_transmission(float3 N, float3 V, float3 X, float2 alpha, int distribution, FresnelData fd, float3 tint) {
    fd.refraction = true;
    if (u_refractionTwoSided) {
        tint = mx_square(tint);
    }
    return mx_environment_radiance(N, V, X, alpha, distribution, fd) * tint;
}

// Light data
struct LightData {
    int type;
    float3 direction;
    float3 color;
    float intensity;
};

cbuffer LightDataBuffer : register(b2) {
    LightData u_lightData[MAX_LIGHT_SOURCES];
};

// Variance shadow occlusion
float mx_variance_shadow_occlusion(float2 moments, float fragmentDepth) {
    const float MIN_VARIANCE = 0.00001;
    float p = (fragmentDepth <= moments.x) ? 1.0 : 0.0;
    float variance = moments.y - mx_square(moments.x);
    variance = max(variance, MIN_VARIANCE);
    float d = fragmentDepth - moments.x;
    float pMax = variance / (variance + mx_square(d));
    return max(p, pMax);
}

float2 mx_compute_depth_moments(float depth) {
    return float2(depth, mx_square(depth));
}

void mx_directional_light(LightData light, float3 position, out LightShader result) {
    result.direction = -light.direction;
    result.intensity = light.color * light.intensity;
}

int numActiveLightSources() {
    return min(u_numActiveLightSources, MAX_LIGHT_SOURCES);
}

void sampleLightSource(LightData light, float3 position, out LightShader result) {
    result.intensity = 0.0;
    result.direction = 0.0;
    if (light.type == 1) {
        mx_directional_light(light, position, result);
    }
}

// Closure data
#define CLOSURE_TYPE_DEFAULT 0
#define CLOSURE_TYPE_REFLECTION 1
#define CLOSURE_TYPE_TRANSMISSION 2
#define CLOSURE_TYPE_INDIRECT 3
#define CLOSURE_TYPE_EMISSION 4

struct ClosureData {
    int closureType;
    float3 L;
    float3 V;
    float3 N;
    float3 P;
    float occlusion;
};

// Imageworks sheen
float mx_imageworks_sheen_NDF(float NdotH, float roughness) {
    float invRoughness = 1.0 / max(roughness, 0.005);
    float cos2 = NdotH * NdotH;
    float sin2 = 1.0 - cos2;
    return (2.0 + invRoughness) * pow(sin2, invRoughness * 0.5) / (2.0 * M_PI);
}

float mx_imageworks_sheen_brdf(float NdotL, float NdotV, float NdotH, float roughness) {
    float D = mx_imageworks_sheen_NDF(NdotH, roughness);
    float F = 1.0;
    float G = 1.0;
    return D * F * G / (4.0 * (NdotL + NdotV - NdotL * NdotV));
}

float mx_imageworks_sheen_dir_albedo_analytic(float NdotV, float roughness) {
    float2 r = float2(13.67300, 1.0) +
               float2(-68.78018, 61.57746) * NdotV +
               float2(799.08825, 442.78211) * roughness +
               float2(-905.00061, 2597.49308) * NdotV * roughness +
               float2(60.28956, 121.81241) * mx_square(NdotV) +
               float2(1086.96473, 3045.55075) * mx_square(roughness);
    return r.x / r.y;
}

float mx_imageworks_sheen_dir_albedo_monte_carlo(float NdotV, float roughness) {
    NdotV = clamp(NdotV, M_FLOAT_EPS, 1.0);
    float3 V = float3(sqrt(1.0 - mx_square(NdotV)), 0, NdotV);
    float radiance = 0.0;
    const int SAMPLE_COUNT = 64;
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float2 Xi = mx_spherical_fibonacci(i, SAMPLE_COUNT);
        float3 L = mx_uniform_sample_hemisphere(Xi);
        float3 H = normalize(L + V);
        float NdotL = clamp(L.z, M_FLOAT_EPS, 1.0);
        float NdotH = clamp(H.z, M_FLOAT_EPS, 1.0);
        float reflectance = mx_imageworks_sheen_brdf(NdotL, NdotV, NdotH, roughness);
        radiance += reflectance * NdotL * 2.0 * M_PI;
    }
    return radiance / float(SAMPLE_COUNT);
}

float mx_imageworks_sheen_dir_albedo(float NdotV, float roughness) {
    #if DIRECTIONAL_ALBEDO_METHOD == 0
        float dirAlbedo = mx_imageworks_sheen_dir_albedo_analytic(NdotV, roughness);
    #else
        float dirAlbedo = mx_imageworks_sheen_dir_albedo_monte_carlo(NdotV, roughness);
    #endif
    return clamp(dirAlbedo, 0.0, 1.0);
}

// Zeltner sheen
float mx_zeltner_sheen_dir_albedo(float x, float y) {
    float s = y * (0.0206607 + 1.58491 * y) / (0.0379424 + y * (1.32227 + y));
    float m = y * (-0.193854 + y * (-1.14885 + y * (1.7932 - 0.95943 * y * y))) / (0.046391 + y);
    float o = y * (0.000654023 + (-0.0207818 + 0.119681 * y) * y) / (1.26264 + y * (-1.92021 + y));
    return exp(-0.5 * mx_square((x - m) / s)) / (s * sqrt(2.0 * M_PI)) + o;
}

float mx_zeltner_sheen_ltc_aInv(float x, float y) {
    return (2.58126 * x + 0.813703 * y) * y / (1.0 + 0.310327 * x * x + 2.60994 * x * y);
}

float mx_zeltner_sheen_ltc_bInv(float x, float y) {
    return sqrt(1.0 - x) * (y - 1.0) * y * y * y / (0.0000254053 + 1.71228 * x - 1.71506 * x * y + 1.34174 * y * y);
}

float3x3 mx_orthonormal_basis_ltc(float3 V, float3 N, float NdotV) {
    float3 X = V - N * NdotV;
    float lenSqr = dot(X, X);
    if (lenSqr > 0.0) {
        X *= mx_inversesqrt(lenSqr);
        float3 Y = cross(N, X);
        return float3x3(X, Y, N);
    }
    return mx_orthonormal_basis(N);
}

float mx_zeltner_sheen_brdf(float3 L, float3 V, float3 N, float NdotV, float roughness) {
    float3x3 toLTC = transpose(mx_orthonormal_basis_ltc(V, N, NdotV));
    float3 w = mul(toLTC, L);
    float aInv = mx_zeltner_sheen_ltc_aInv(NdotV, roughness);
    float bInv = mx_zeltner_sheen_ltc_bInv(NdotV, roughness);
    float3 wo = float3(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
    float lenSqr = dot(wo, wo);
    return max(wo.z, 0.0) * M_PI_INV * mx_square(aInv / lenSqr);
}

float3 mx_zeltner_sheen_importance_sample(float2 Xi, float3 V, float3 N, float roughness, out float pdf) {
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    roughness = clamp(roughness, 0.01, 1.0);
    float3 wo = mx_cosine_sample_hemisphere(Xi);
    float aInv = mx_zeltner_sheen_ltc_aInv(NdotV, roughness);
    float bInv = mx_zeltner_sheen_ltc_bInv(NdotV, roughness);
    float3 w = float3(wo.x / aInv - wo.z * bInv / aInv, wo.y / aInv, wo.z);
    float lenSqr = dot(w, w);
    w *= mx_inversesqrt(lenSqr);
    pdf = max(w.z, 0.0) * M_PI_INV * mx_square(aInv * lenSqr);
    float3x3 fromLTC = mx_orthonormal_basis_ltc(V, N, NdotV);
    w = mul(fromLTC, w);
    return w;
}

void mx_sheen_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N, int mode, inout BSDF bsdf) {
    if (weight < M_FLOAT_EPS) {
        return;
    }
    float3 V = closureData.V;
    float3 L = closureData.L;
    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    if (closureData.closureType == CLOSURE_TYPE_REFLECTION) {
        float dirAlbedo;
        if (mode == 0) {
            float3 H = normalize(L + V);
            float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
            float NdotH = clamp(dot(N, H), M_FLOAT_EPS, 1.0);
            float3 fr = color * mx_imageworks_sheen_brdf(NdotL, NdotV, NdotH, roughness);
            dirAlbedo = mx_imageworks_sheen_dir_albedo(NdotV, roughness);
            bsdf.response = fr * NdotL * closureData.occlusion * weight;
        } else {
            roughness = clamp(roughness, 0.01, 1.0);
            float3 fr = color * mx_zeltner_sheen_brdf(L, V, N, NdotV, roughness);
            dirAlbedo = mx_zeltner_sheen_dir_albedo(NdotV, roughness);
            bsdf.response = dirAlbedo * fr * closureData.occlusion * weight;
        }
        bsdf.throughput = 1.0 - dirAlbedo * weight;
    } else if (closureData.closureType == CLOSURE_TYPE_INDIRECT) {
        float dirAlbedo;
        if (mode == 0) {
            dirAlbedo = mx_imageworks_sheen_dir_albedo(NdotV, roughness);
        } else {
            roughness = clamp(roughness, 0.01, 1.0);
            dirAlbedo = mx_zeltner_sheen_dir_albedo(NdotV, roughness);
        }
        float3 Li = mx_environment_irradiance(N);
        bsdf.response = Li * color * dirAlbedo * weight;
        bsdf.throughput = 1.0 - dirAlbedo * weight;
    }
}

void NG_open_pbr_anisotropy(float roughness, float anisotropy, out float2 out1) {
    float rough_sq_out = roughness * roughness;
    const float aniso_invert_amount_tmp = 1.0;
    float aniso_invert_out = aniso_invert_amount_tmp - anisotropy;
    float aniso_invert_sq_out = aniso_invert_out * aniso_invert_out;
    const float denom_in2_tmp = 1.0;
    float denom_out = aniso_invert_sq_out + denom_in2_tmp;
    const float fraction_in1_tmp = 2.0;
    float fraction_out = fraction_in1_tmp * aniso_invert_out / denom_out;
    out1 = float2(rough_sq_out * fraction_out, rough_sq_out / fraction_out);
}

void NG_convert_float_color3(float in1, out float3 out1) {
    out1 = float3(in1, in1, in1);
}

void mx_generalized_schlick_edf(ClosureData closureData, float3 color0, float3 color90, float exponent, EDF base, out EDF result) {
    if (closureData.closureType == CLOSURE_TYPE_EMISSION) {
        float3 N = mx_forward_facing_normal(closureData.N, closureData.V);
        float NdotV = clamp(dot(N, closureData.V), M_FLOAT_EPS, 1.0);
        float3 f = mx_fresnel_schlick(NdotV, color0, color90, exponent);
        result = base * f;
    }
}

void mx_mix_edf(ClosureData closureData, EDF fg, EDF bg, float mixValue, out EDF result) {
    result = lerp(bg, fg, mixValue);
}

void mx_generalized_schlick_bsdf(ClosureData closureData, float weight, float3 color0, float3 color82, float3 color90, float exponent, float2 roughness, float thinfilm_thickness, float thinfilm_ior, float3 N, float3 X, int distribution, int scatter_mode, inout BSDF bsdf) {
    if (weight < M_FLOAT_EPS) {
        return;
    }
    if (closureData.closureType != CLOSURE_TYPE_TRANSMISSION && scatter_mode == 1) {
        return;
    }
    float3 V = closureData.V;
    float3 L = closureData.L;
    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    float3 safeColor0 = max(color0, 0.0);
    float3 safeColor82 = max(color82, 0.0);
    float3 safeColor90 = max(color90, 0.0);
    FresnelData fd = mx_init_fresnel_schlick(safeColor0, safeColor82, safeColor90, exponent, thinfilm_thickness, thinfilm_ior);
    float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);
    if (closureData.closureType == CLOSURE_TYPE_REFLECTION) {
        X = normalize(X - dot(X, N) * N);
        float3 Y = cross(N, X);
        float3 H = normalize(L + V);
        float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
        float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);
        float3 Ht = float3(dot(H, X), dot(H, Y), dot(H, N));
        float3 F = mx_compute_fresnel(VdotH, fd);
        float D = mx_ggx_NDF(Ht, safeAlpha);
        float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);
        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;
        float avgDirAlbedo = dot(dirAlbedo, 1.0 / 3.0);
        bsdf.throughput = 1.0 - avgDirAlbedo * weight;
        bsdf.response = D * F * G * comp * closureData.occlusion * weight / (4.0 * NdotV);
    } else if (closureData.closureType == CLOSURE_TYPE_TRANSMISSION) {
        float3 F = mx_compute_fresnel(NdotV, fd);
        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;
        float avgDirAlbedo = dot(dirAlbedo, 1.0 / 3.0);
        bsdf.throughput = 1.0 - avgDirAlbedo * weight;
        if (scatter_mode != 0) {
            float avgF0 = dot(safeColor0, 1.0 / 3.0);
            fd.ior = float3(mx_f0_to_ior(avgF0).xxx);
            bsdf.response = mx_surface_transmission(N, V, X, safeAlpha, distribution, fd, 1.0) * weight;
        }
    } else if (closureData.closureType == CLOSURE_TYPE_INDIRECT) {
        float3 F = mx_compute_fresnel(NdotV, fd);
        float3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);
        float3 dirAlbedo = mx_ggx_dir_albedo(NdotV, avgAlpha, safeColor0, safeColor90) * comp;
        float avgDirAlbedo = dot(dirAlbedo, 1.0 / 3.0);
        bsdf.throughput = 1.0 - avgDirAlbedo * weight;
        float3 Li = mx_environment_radiance(N, V, X, safeAlpha, distribution, fd);
        bsdf.response = Li * comp * weight;
    }
}

void mx_anisotropic_vdf(ClosureData closureData, float3 absorption, float3 scattering, float anisotropy, inout BSDF bsdf) {
    // Placeholder: volumetric light absorption approximation needed
}

void mx_layer_vdf(ClosureData closureData, BSDF top, BSDF base, out BSDF result) {
    result.response = top.response + base.response;
    result.throughput = top.throughput + base.throughput;
}

void mx_layer_bsdf(ClosureData closureData, BSDF top, BSDF base, out BSDF result) {
    result.response = top.response + base.response * top.throughput;
    result.throughput = top.throughput + base.throughput;
}

// ------------------
void mx_mix_bsdf(ClosureData closure, BSDF bsdf1, BSDF bsdf2, float weight, out BSDF bsdf)
{
    bsdf.response = lerp(bsdf1.response, bsdf2.response, weight);
    bsdf.throughput = lerp(bsdf1.throughput, bsdf2.throughput, weight); // Placeholder
}

void mx_dielectric_bsdf(ClosureData closure, float weight, float3 color, float ior, float2 roughness, float tf_thickness, float tf_ior, float3 normal, float3 tangent, int flag1, int flag2, out BSDF bsdf)
{
    bsdf.response = float3(0, 0, 0);
    bsdf.throughput = float3(1, 1, 1); // Placeholder
}

void mx_multiply_bsdf_color3(ClosureData closure, BSDF bsdf, float3 color, out BSDF bsdf_out)
{
    bsdf_out.response = bsdf.response * color;
    bsdf_out.throughput = bsdf.throughput * color; // Placeholder
}

void mx_oren_nayar_diffuse_bsdf(ClosureData closure, float weight, float3 color, float roughness, float3 normal, bool flag, out BSDF bsdf)
{
    bsdf.response = float3(0, 0, 0);
    bsdf.throughput = float3(1, 1, 1); // Placeholder
}

void mx_subsurface_bsdf(ClosureData closure, float weight, float3 color, float3 radius, float anisotropy, float3 normal, out BSDF bsdf)
{
    bsdf.response = float3(0, 0, 0);
    bsdf.throughput = float3(1, 1, 1); // Placeholder
}

void mx_translucent_bsdf(ClosureData closure, float weight, float3 color, float3 normal, out BSDF bsdf)
{
    bsdf.response = float3(0, 0, 0);
    bsdf.throughput = float3(1, 1, 1); // Placeholder
}

void mx_multiply_edf_color3(ClosureData closure, EDF edf, float3 color, out EDF edf_out)
{
    edf_out = edf * color; // Placeholder
}

void mx_uniform_edf(ClosureData closure, float3 weight, out EDF edf)
{
    edf = weight; // Placeholder
}

// ------------------

// PBR surface shader function (translated fragment)
void NG_open_pbr_surface_surfaceshader_fragment(
    float base_weight,
    float3 base_color,
    float base_diffuse_roughness,
    float base_metalness,
    float specular_weight,
    float3 specular_color,
    float specular_roughness,
    float specular_ior,
    float specular_roughness_anisotropy,
    float transmission_weight,
    float3 transmission_color,
    float transmission_depth,
    float3 transmission_scatter,
    float transmission_scatter_anisotropy,
    float transmission_dispersion_scale,
    float transmission_dispersion_abbe_number,
    float subsurface_weight,
    float3 subsurface_color,
    float subsurface_radius,
    float3 subsurface_radius_scale,
    float subsurface_scatter_anisotropy,
    float fuzz_weight,
    float3 fuzz_color,
    float fuzz_roughness,
    float coat_weight,
    float3 coat_color,
    float coat_roughness,
    float coat_roughness_anisotropy,
    float coat_ior,
    float coat_darkening,
    float thin_film_weight,
    float thin_film_thickness,
    float thin_film_ior,
    float emission_luminance,
    float3 emission_color,
    float geometry_opacity,
    bool geometry_thin_walled,
    float3 geometry_normal,
    float3 geometry_coat_normal,
    float3 geometry_tangent,
    float3 geometry_coat_tangent,
    float3 metal_reflectivity_out,
    float3 metal_edgecolor_out,
    float2 coat_roughness_vector_out,
    float coat_roughness_to_power_4_out,
    float specular_roughness_to_power_4_out,
    float thin_film_thickness_nm_out,
    float specular_to_coat_ior_ratio_out,
    float coat_to_specular_ior_ratio_out,
    float3 if_transmission_tint_out,
    float3 transmission_color_vector_out,
    float3 transmission_depth_vector_out,
    float3 transmission_scatter_vector_out,
    float3 subsurface_color_nonnegative_out,
    float one_minus_subsurface_scatter_anisotropy_out,
    float one_plus_subsurface_scatter_anisotropy_out,
    float3 subsurface_radius_scaled_out,
    float subsurface_selector_out,
    float3 base_color_nonnegative_out,
    float coat_ior_minus_one_out,
    float coat_ior_plus_one_out,
    float coat_ior_sqr_out,
    float3 Emetal_out,
    float3 Edielectric_out,
    float coat_weight_times_coat_darkening_out,
    float3 coat_attenuation_out,
    float3 emission_weight_out,
    float two_times_coat_roughness_to_power_4_out,
    float specular_to_coat_ior_ratio_tir_fix_out,
    float3 transmission_color_ln_out,
    float3 scattering_coeff_out,
    float3 subsurface_thin_walled_brdf_factor_out,
    float3 subsurface_thin_walled_btdf_factor_out,
    float coat_ior_to_F0_sqrt_out,
    float3 Ebase_out,
    float add_coat_and_spec_roughnesses_to_power_4_out,
    float eta_s_out,
    float3 extinction_coeff_denom_out,
    float3 if_volume_scattering_out,
    float coat_ior_to_F0_out,
    float min_1_add_coat_and_spec_roughnesses_to_power_4_out,
    float eta_s_minus_one_out,
    float eta_s_plus_one_out,
    float3 extinction_coeff_out,
    float one_minus_coat_F0_out,
    PSInput input,
    out SurfaceShader out1)
{
    const float coat_affected_specular_roughness_in2_tmp = 0.25;
    float coat_affected_specular_roughness_out = pow(min_1_add_coat_and_spec_roughnesses_to_power_4_out, coat_affected_specular_roughness_in2_tmp);

    float sign_eta_s_minus_one_out = sign(eta_s_minus_one_out);

    float specular_F0_sqrt_out = eta_s_minus_one_out / eta_s_plus_one_out;

    float3 absorption_coeff_out = extinction_coeff_out - scattering_coeff_out;

    float one_minus_coat_F0_over_eta2_out = one_minus_coat_F0_out / coat_ior_sqr_out;

    float3 one_minus_coat_F0_color_out = float3(0.0, 0.0, 0.0);
    NG_convert_float_color3(one_minus_coat_F0_out, one_minus_coat_F0_color_out);

    float effective_specular_roughness_out = lerp(specular_roughness, coat_affected_specular_roughness_out, coat_weight);

    float specular_F0_out = specular_F0_sqrt_out * specular_F0_sqrt_out;

    const int absorption_coeff_x_index_tmp = 0;
    float absorption_coeff_x_out = absorption_coeff_out[absorption_coeff_x_index_tmp];

    const int absorption_coeff_y_index_tmp = 1;
    float absorption_coeff_y_out = absorption_coeff_out[absorption_coeff_y_index_tmp];

    const int absorption_coeff_z_index_tmp = 2;
    float absorption_coeff_z_out = absorption_coeff_out[absorption_coeff_z_index_tmp];

    const float Kcoat_in1_tmp = 1.0;
    float Kcoat_out = Kcoat_in1_tmp - one_minus_coat_F0_over_eta2_out;

    float2 main_roughness_out = float2(0.0, 0.0);
    NG_open_pbr_anisotropy(effective_specular_roughness_out, specular_roughness_anisotropy, main_roughness_out);

    float scaled_specular_F0_out = specular_weight * specular_F0_out;

    float absorption_coeff_min_x_y_out = min(absorption_coeff_x_out, absorption_coeff_y_out);

    const float one_minus_Kcoat_in1_tmp = 1.0;
    float one_minus_Kcoat_out = one_minus_Kcoat_in1_tmp - Kcoat_out;

    float3 Ebase_Kcoat_out = Ebase_out * Kcoat_out;

    const float scaled_specular_F0_clamped_low_tmp = 0.0;
    const float scaled_specular_F0_clamped_high_tmp = 0.99999;
    float scaled_specular_F0_clamped_out = clamp(scaled_specular_F0_out, scaled_specular_F0_clamped_low_tmp, scaled_specular_F0_clamped_high_tmp);

    float absorption_coeff_min_out = min(absorption_coeff_min_x_y_out, absorption_coeff_z_out);

    float3 one_minus_Kcoat_color_out = float3(0.0, 0.0, 0.0);
    NG_convert_float_color3(one_minus_Kcoat_out, one_minus_Kcoat_color_out);

    const float3 one_minus_Ebase_Kcoat_in1_tmp = float3(1.0, 1.0, 1.0);
    float3 one_minus_Ebase_Kcoat_out = one_minus_Ebase_Kcoat_in1_tmp - Ebase_Kcoat_out;

    float sqrt_scaled_specular_F0_out = sqrt(scaled_specular_F0_clamped_out);

    float3 absorption_coeff_min_vector_out = float3(0.0, 0.0, 0.0);
    NG_convert_float_color3(absorption_coeff_min_out, absorption_coeff_min_vector_out);

    float3 base_darkening_out = one_minus_Kcoat_color_out / one_minus_Ebase_Kcoat_out;

    float modulated_eta_s_epsilon_out = sign_eta_s_minus_one_out * sqrt_scaled_specular_F0_out;

    float3 absorption_coeff_shifted_out = absorption_coeff_out - absorption_coeff_min_vector_out;

    const float3 modulated_base_darkening_bg_tmp = float3(1.0, 1.0, 1.0);
    float3 modulated_base_darkening_out = lerp(modulated_base_darkening_bg_tmp, base_darkening_out, coat_weight_times_coat_darkening_out);

    const float one_plus_modulated_eta_s_epsilon_in1_tmp = 1.0;
    float one_plus_modulated_eta_s_epsilon_out = one_plus_modulated_eta_s_epsilon_in1_tmp + modulated_eta_s_epsilon_out;

    const float one_minus_modulated_eta_s_epsilon_in1_tmp = 1.0;
    float one_minus_modulated_eta_s_epsilon_out = one_minus_modulated_eta_s_epsilon_in1_tmp - modulated_eta_s_epsilon_out;

    const float if_absorption_coeff_shifted_value1_tmp = 0.0;
    float3 if_absorption_coeff_shifted_out = (if_absorption_coeff_shifted_value1_tmp > absorption_coeff_min_out) ? absorption_coeff_shifted_out : absorption_coeff_out;

    float modulated_eta_s_out = one_plus_modulated_eta_s_epsilon_out / one_minus_modulated_eta_s_epsilon_out;

    const float if_volume_absorption_value2_tmp = 0.0;
    const float3 if_volume_absorption_in2_tmp = float3(0.0, 0.0, 0.0);
    float3 if_volume_absorption_out = (transmission_depth > if_volume_absorption_value2_tmp) ? if_absorption_coeff_shifted_out : if_volume_absorption_in2_tmp;

    SurfaceShader shader_constructor_out = { float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0) };
    {
        float3 N = normalize(input.normalWorld);
        float3 V = normalize(u_viewPosition.xyz - input.positionWorld);
        float3 P = input.positionWorld;
        float3 L = float3(0.0, 0.0, 0.0);
        float occlusion = 1.0;

        float surfaceOpacity = geometry_opacity;

        // Shadow occlusion
        float3 shadowCoord = mul(u_shadowMatrix, float4(input.positionWorld, 1.0)).xyz;
        shadowCoord = shadowCoord * 0.5 + 0.5;
        float2 shadowMoments = u_shadowMap.Sample(u_shadowSampler, shadowCoord.xy).xy;
        occlusion = mx_variance_shadow_occlusion(shadowMoments, shadowCoord.z);

        // Light loop
        int numLights = numActiveLightSources();
        LightShader lightShader;
        [loop]
        for (int activeLightIndex = 0; activeLightIndex < numLights; ++activeLightIndex)
        {
            sampleLightSource(u_lightData[activeLightIndex], input.positionWorld, lightShader);
            L = lightShader.direction;

            // Calculate the BSDF response for this light source
            ClosureData closureData = { CLOSURE_TYPE_REFLECTION, L, V, N, P, occlusion };
            BSDF fuzz_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_sheen_bsdf(closureData, fuzz_weight, fuzz_color, fuzz_roughness, geometry_normal, 1, fuzz_bsdf_out);

            BSDF coat_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, coat_weight, float3(1.0, 1.0, 1.0), coat_ior, coat_roughness_vector_out, 0.0, 1.5, geometry_coat_normal, geometry_coat_tangent, 0, 0, coat_bsdf_out);

            BSDF metal_bsdf_tf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_generalized_schlick_bsdf(closureData, specular_weight, metal_reflectivity_out, metal_edgecolor_out, float3(1.0, 1.0, 1.0), 5.0, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, 0, 0, metal_bsdf_tf_out);

            BSDF metal_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_generalized_schlick_bsdf(closureData, specular_weight, metal_reflectivity_out, metal_edgecolor_out, float3(1.0, 1.0, 1.0), 5.0, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 0, metal_bsdf_out);

            BSDF metal_bsdf_tf_mix_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, metal_bsdf_tf_out, metal_bsdf_out, thin_film_weight, metal_bsdf_tf_mix_out);

            BSDF dielectric_reflection_tf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, specular_color, modulated_eta_s_out, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, 0, 0, dielectric_reflection_tf_out);

            BSDF dielectric_reflection_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, specular_color, modulated_eta_s_out, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 0, dielectric_reflection_out);

            BSDF dielectric_reflection_tf_mix_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, dielectric_reflection_tf_out, dielectric_reflection_out, thin_film_weight, dielectric_reflection_tf_mix_out);

            BSDF dielectric_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, if_transmission_tint_out, modulated_eta_s_out, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 1, dielectric_transmission_out);

            BSDF dielectric_volume_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_anisotropic_vdf(closureData, if_volume_absorption_out, if_volume_scattering_out, transmission_scatter_anisotropy, dielectric_volume_out);

            BSDF dielectric_volume_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_vdf(closureData, dielectric_transmission_out, dielectric_volume_out, dielectric_volume_transmission_out);

            BSDF subsurface_thin_walled_reflection_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_oren_nayar_diffuse_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, base_diffuse_roughness, geometry_normal, false, subsurface_thin_walled_reflection_bsdf_out);

            BSDF subsurface_thin_walled_reflection_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, subsurface_thin_walled_reflection_bsdf_out, subsurface_thin_walled_brdf_factor_out, subsurface_thin_walled_reflection_out);

            BSDF subsurface_thin_walled_transmission_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_translucent_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, geometry_normal, subsurface_thin_walled_transmission_bsdf_out);

            BSDF subsurface_thin_walled_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, subsurface_thin_walled_transmission_bsdf_out, subsurface_thin_walled_btdf_factor_out, subsurface_thin_walled_transmission_out);

            BSDF subsurface_thin_walled_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, subsurface_thin_walled_reflection_out, subsurface_thin_walled_transmission_out, 0.5, subsurface_thin_walled_out);

            BSDF subsurface_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_subsurface_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, subsurface_radius_scaled_out, subsurface_scatter_anisotropy, geometry_normal, subsurface_bsdf_out);

            BSDF selected_subsurface_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, subsurface_thin_walled_out, subsurface_bsdf_out, subsurface_selector_out, selected_subsurface_out);

            BSDF diffuse_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_oren_nayar_diffuse_bsdf(closureData, base_weight, base_color_nonnegative_out, base_diffuse_roughness, geometry_normal, true, diffuse_bsdf_out);

            BSDF opaque_base_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, selected_subsurface_out, diffuse_bsdf_out, subsurface_weight, opaque_base_out);

            BSDF dielectric_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, dielectric_volume_transmission_out, opaque_base_out, transmission_weight, dielectric_substrate_out);

            BSDF dielectric_base_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, dielectric_reflection_tf_mix_out, dielectric_substrate_out, dielectric_base_out);

            BSDF base_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, metal_bsdf_tf_mix_out, dielectric_base_out, base_metalness, base_substrate_out);

            BSDF darkened_base_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, base_substrate_out, modulated_base_darkening_out, darkened_base_substrate_out);

            BSDF coat_substrate_attenuated_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, darkened_base_substrate_out, coat_attenuation_out, coat_substrate_attenuated_out);

            BSDF coat_layer_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, coat_bsdf_out, coat_substrate_attenuated_out, coat_layer_out);

            BSDF fuzz_layer_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, fuzz_bsdf_out, coat_layer_out, fuzz_layer_out);

            // Accumulate the light's contribution
            shader_constructor_out.color += lightShader.intensity * fuzz_layer_out.response;
        }

        // Ambient occlusion
        occlusion = 1.0;

        // Add environment contribution
        {
            ClosureData closureData = { CLOSURE_TYPE_INDIRECT, L, V, N, P, occlusion };
            BSDF fuzz_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_sheen_bsdf(closureData, fuzz_weight, fuzz_color, fuzz_roughness, geometry_normal, 1, fuzz_bsdf_out);

            BSDF coat_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, coat_weight, float3(1.0, 1.0, 1.0), coat_ior, coat_roughness_vector_out, 0.0, 1.5, geometry_coat_normal, geometry_coat_tangent, 0, 0, coat_bsdf_out);

            BSDF metal_bsdf_tf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_generalized_schlick_bsdf(closureData, specular_weight, metal_reflectivity_out, metal_edgecolor_out, float3(1.0, 1.0, 1.0), 5.0, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, 0, 0, metal_bsdf_tf_out);

            BSDF metal_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_generalized_schlick_bsdf(closureData, specular_weight, metal_reflectivity_out, metal_edgecolor_out, float3(1.0, 1.0, 1.0), 5.0, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 0, metal_bsdf_out);

            BSDF metal_bsdf_tf_mix_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, metal_bsdf_tf_out, metal_bsdf_out, thin_film_weight, metal_bsdf_tf_mix_out);

            BSDF dielectric_reflection_tf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, specular_color, modulated_eta_s_out, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, 0, 0, dielectric_reflection_tf_out);

            BSDF dielectric_reflection_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, specular_color, modulated_eta_s_out, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 0, dielectric_reflection_out);

            BSDF dielectric_reflection_tf_mix_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, dielectric_reflection_tf_out, dielectric_reflection_out, thin_film_weight, dielectric_reflection_tf_mix_out);

            BSDF dielectric_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, if_transmission_tint_out, modulated_eta_s_out, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 1, dielectric_transmission_out);

            BSDF dielectric_volume_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_anisotropic_vdf(closureData, if_volume_absorption_out, if_volume_scattering_out, transmission_scatter_anisotropy, dielectric_volume_out);

            BSDF dielectric_volume_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_vdf(closureData, dielectric_transmission_out, dielectric_volume_out, dielectric_volume_transmission_out);

            BSDF subsurface_thin_walled_reflection_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_oren_nayar_diffuse_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, base_diffuse_roughness, geometry_normal, false, subsurface_thin_walled_reflection_bsdf_out);

            BSDF subsurface_thin_walled_reflection_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, subsurface_thin_walled_reflection_bsdf_out, subsurface_thin_walled_brdf_factor_out, subsurface_thin_walled_reflection_out);

            BSDF subsurface_thin_walled_transmission_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_translucent_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, geometry_normal, subsurface_thin_walled_transmission_bsdf_out);

            BSDF subsurface_thin_walled_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, subsurface_thin_walled_transmission_bsdf_out, subsurface_thin_walled_btdf_factor_out, subsurface_thin_walled_transmission_out);

            BSDF subsurface_thin_walled_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, subsurface_thin_walled_reflection_out, subsurface_thin_walled_transmission_out, 0.5, subsurface_thin_walled_out);

            BSDF subsurface_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_subsurface_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, subsurface_radius_scaled_out, subsurface_scatter_anisotropy, geometry_normal, subsurface_bsdf_out);

            BSDF selected_subsurface_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, subsurface_thin_walled_out, subsurface_bsdf_out, subsurface_selector_out, selected_subsurface_out);

            BSDF diffuse_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_oren_nayar_diffuse_bsdf(closureData, base_weight, base_color_nonnegative_out, base_diffuse_roughness, geometry_normal, true, diffuse_bsdf_out);

            BSDF opaque_base_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, selected_subsurface_out, diffuse_bsdf_out, subsurface_weight, opaque_base_out);

            BSDF dielectric_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, dielectric_volume_transmission_out, opaque_base_out, transmission_weight, dielectric_substrate_out);

            BSDF dielectric_base_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, dielectric_reflection_tf_mix_out, dielectric_substrate_out, dielectric_base_out);

            BSDF base_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, metal_bsdf_tf_mix_out, dielectric_base_out, base_metalness, base_substrate_out);

            BSDF darkened_base_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, base_substrate_out, modulated_base_darkening_out, darkened_base_substrate_out);

            BSDF coat_substrate_attenuated_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, darkened_base_substrate_out, coat_attenuation_out, coat_substrate_attenuated_out);

            BSDF coat_layer_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, coat_bsdf_out, coat_substrate_attenuated_out, coat_layer_out);

            BSDF fuzz_layer_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, fuzz_bsdf_out, coat_layer_out, fuzz_layer_out);

            shader_constructor_out.color += occlusion * fuzz_layer_out.response;
        }

        // Add surface emission
        {
            ClosureData closureData = { CLOSURE_TYPE_EMISSION, L, V, N, P, occlusion };
            EDF uncoated_emission_edf_out = { float3(0.0, 0.0, 0.0) };
            mx_uniform_edf(closureData, emission_weight_out, uncoated_emission_edf_out);

            EDF coat_tinted_emission_edf_out = { float3(0.0, 0.0, 0.0) };
            mx_multiply_edf_color3(closureData, uncoated_emission_edf_out, coat_color, coat_tinted_emission_edf_out);

            EDF coated_emission_edf_out = { float3(0.0, 0.0, 0.0) };
            mx_generalized_schlick_edf(closureData, one_minus_coat_F0_color_out, float3(0.0, 0.0, 0.0), 5.0, coat_tinted_emission_edf_out, coated_emission_edf_out);

            EDF emission_edf_out = { float3(0.0, 0.0, 0.0) };
            mx_mix_edf(closureData, coated_emission_edf_out, uncoated_emission_edf_out, coat_weight, emission_edf_out);

            shader_constructor_out.color += emission_edf_out;
        }

        // Calculate the BSDF transmission for viewing direction
        {
            ClosureData closureData = { CLOSURE_TYPE_TRANSMISSION, L, V, N, P, occlusion };
            BSDF fuzz_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_sheen_bsdf(closureData, fuzz_weight, fuzz_color, fuzz_roughness, geometry_normal, 1, fuzz_bsdf_out);

            BSDF coat_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, coat_weight, float3(1.0, 1.0, 1.0), coat_ior, coat_roughness_vector_out, 0.0, 1.5, geometry_coat_normal, geometry_coat_tangent, 0, 0, coat_bsdf_out);

            BSDF metal_bsdf_tf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_generalized_schlick_bsdf(closureData, specular_weight, metal_reflectivity_out, metal_edgecolor_out, float3(1.0, 1.0, 1.0), 5.0, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, 0, 0, metal_bsdf_tf_out);

            BSDF metal_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_generalized_schlick_bsdf(closureData, specular_weight, metal_reflectivity_out, metal_edgecolor_out, float3(1.0, 1.0, 1.0), 5.0, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 0, metal_bsdf_out);

            BSDF metal_bsdf_tf_mix_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, metal_bsdf_tf_out, metal_bsdf_out, thin_film_weight, metal_bsdf_tf_mix_out);

            BSDF dielectric_reflection_tf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, specular_color, modulated_eta_s_out, main_roughness_out, thin_film_thickness_nm_out, thin_film_ior, geometry_normal, geometry_tangent, 0, 0, dielectric_reflection_tf_out);

            BSDF dielectric_reflection_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, specular_color, modulated_eta_s_out, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 0, dielectric_reflection_out);

            BSDF dielectric_reflection_tf_mix_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, dielectric_reflection_tf_out, dielectric_reflection_out, thin_film_weight, dielectric_reflection_tf_mix_out);

            BSDF dielectric_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_dielectric_bsdf(closureData, 1.0, if_transmission_tint_out, modulated_eta_s_out, main_roughness_out, 0.0, 1.5, geometry_normal, geometry_tangent, 0, 1, dielectric_transmission_out);

            BSDF dielectric_volume_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_anisotropic_vdf(closureData, if_volume_absorption_out, if_volume_scattering_out, transmission_scatter_anisotropy, dielectric_volume_out);

            BSDF dielectric_volume_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_vdf(closureData, dielectric_transmission_out, dielectric_volume_out, dielectric_volume_transmission_out);

            BSDF subsurface_thin_walled_reflection_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_oren_nayar_diffuse_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, base_diffuse_roughness, geometry_normal, false, subsurface_thin_walled_reflection_bsdf_out);

            BSDF subsurface_thin_walled_reflection_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, subsurface_thin_walled_reflection_bsdf_out, subsurface_thin_walled_brdf_factor_out, subsurface_thin_walled_reflection_out);

            BSDF subsurface_thin_walled_transmission_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_translucent_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, geometry_normal, subsurface_thin_walled_transmission_bsdf_out);

            BSDF subsurface_thin_walled_transmission_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, subsurface_thin_walled_transmission_bsdf_out, subsurface_thin_walled_btdf_factor_out, subsurface_thin_walled_transmission_out);

            BSDF subsurface_thin_walled_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, subsurface_thin_walled_reflection_out, subsurface_thin_walled_transmission_out, 0.5, subsurface_thin_walled_out);

            BSDF subsurface_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_subsurface_bsdf(closureData, 1.0, subsurface_color_nonnegative_out, subsurface_radius_scaled_out, subsurface_scatter_anisotropy, geometry_normal, subsurface_bsdf_out);

            BSDF selected_subsurface_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, subsurface_thin_walled_out, subsurface_bsdf_out, subsurface_selector_out, selected_subsurface_out);

            BSDF diffuse_bsdf_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_oren_nayar_diffuse_bsdf(closureData, base_weight, base_color_nonnegative_out, base_diffuse_roughness, geometry_normal, true, diffuse_bsdf_out);

            BSDF opaque_base_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, selected_subsurface_out, diffuse_bsdf_out, subsurface_weight, opaque_base_out);

            BSDF dielectric_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, dielectric_volume_transmission_out, opaque_base_out, transmission_weight, dielectric_substrate_out);

            BSDF dielectric_base_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, dielectric_reflection_tf_mix_out, dielectric_substrate_out, dielectric_base_out);

            BSDF base_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_mix_bsdf(closureData, metal_bsdf_tf_mix_out, dielectric_base_out, base_metalness, base_substrate_out);

            BSDF darkened_base_substrate_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, base_substrate_out, modulated_base_darkening_out, darkened_base_substrate_out);

            BSDF coat_substrate_attenuated_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_multiply_bsdf_color3(closureData, darkened_base_substrate_out, coat_attenuation_out, coat_substrate_attenuated_out);

            BSDF coat_layer_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, coat_bsdf_out, coat_substrate_attenuated_out, coat_layer_out);

            BSDF fuzz_layer_out = { float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0) };
            mx_layer_bsdf(closureData, fuzz_bsdf_out, coat_layer_out, fuzz_layer_out);

            shader_constructor_out.color += fuzz_layer_out.response;
        }

        // Compute and apply surface opacity
        {
            shader_constructor_out.color *= surfaceOpacity;
            shader_constructor_out.transparency = lerp(float3(1.0, 1.0, 1.0), shader_constructor_out.transparency, surfaceOpacity);
        }
    }

    out1 = shader_constructor_out;
}

float4 main(PSInput input) : SV_Target
{
    float3 geomprop_Nworld_out1 = normalize(input.normalWorld);
    float3 geomprop_Tworld_out1 = normalize(input.tangentWorld);

    SurfaceShader open_pbr_surface_surfaceshader_out = { float3(0, 0, 0), float3(0, 0, 0) };

    SurfaceShader Default_out = open_pbr_surface_surfaceshader_out;
    return float4(Default_out.color, 1.0);
}























            