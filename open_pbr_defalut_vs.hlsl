cbuffer PrivateUniforms : register(b0)
{
    matrix u_worldMatrix;
    matrix u_viewProjectionMatrix;
    matrix u_worldInverseTransposeMatrix;
};

// Input structure
struct VertexInput
{
    float3 i_position : POSITION;
    float3 i_normal : NORMAL;
    float3 i_tangent : TANGENT;
};

// Output structure
struct VertexOutput
{
    float4 position : SV_POSITION;
    float3 normalWorld : TEXCOORD0;
    float3 tangentWorld : TEXCOORD1;
    float3 positionWorld : TEXCOORD2;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    
    // Transform position to world space and then to clip space
    float4 hPositionWorld = mul(u_worldMatrix, float4(input.i_position, 1.0));
    output.position = mul(u_viewProjectionMatrix, hPositionWorld);
    
    // Transform normal and tangent to world space
    output.normalWorld = normalize(mul((float3x3)u_worldInverseTransposeMatrix, input.i_normal));
    output.tangentWorld = normalize(mul((float3x3)u_worldMatrix, input.i_tangent));
    
    // Pass world position
    output.positionWorld = hPositionWorld.xyz;
    
    return output;
}