const float ROOT_3 = GLSL_sqrt(3.0f);

float3 hexCoord(float3 p, float hexSize) {
    float3 scaledP = p / hexSize;
    float x = ((ROOT_3 * scaledP.x) - scaledP.y - (scaledP.z * 2.0f / 3.0f)) / 3.0f;
    float y = (2.0f * (scaledP.y - (scaledP.z / 3.0f))) / 3.0f;
    float z = (2.0f * scaledP.z) / 3.0f;
    float rx = round(x);
    float ry = round(y);
    float rz = round(z);
    float xDiff = rx - x;
    float yDiff = ry - y;
    float zDiff = rz - z;
    float xShift = GLSL_abs(xDiff * 2.0f + yDiff + zDiff);
    float yShift = GLSL_abs(xDiff + yDiff * 2.0f + zDiff);
    if (xShift > 1.0f || yShift > 1.0f) {
        if (xShift > yShift) {
            rx -= GLSL_sign(xDiff);
        }
        else {
            ry -= GLSL_sign(yDiff);
        }
    }
    return (float3)(rx, ry, rz);
}

float3 offsetCoord(float3 coord, float agitation) {
    float3 d = (float3)(GLSL_dot(coord, (float3)(123.1f, 311.7f, 741.7f)), GLSL_dot(coord, (float3)(269.7f, 183.3f, 317.9f)), GLSL_dot(coord, (float3)(147.3f, 292.1f, 457.1f)));
    float3 f = GLSL_fract(GLSL_sin(d) * 43758.5453f) - 0.5f;
    return coord + f * agitation;
}

float3 unskew(float3 hexCoord, float hexSize) {
    float3 scaledHexCenter = (float3)((hexCoord.x + ((hexCoord.y + hexCoord.z) * 0.5f)) * ROOT_3, (hexCoord.y * 1.5f) + (hexCoord.z * 0.5f), hexCoord.z * 1.5f);
    return scaledHexCenter * hexSize;
}

float hexDistance(float3 p, float3 center, float hexSize) {
    return GLSL_length(p - center) / hexSize;
}

float2 minDistances(float3 p, float3 hexCoord, float3 neighborOffset, float2 minDists, float hexSize, float agitation) {
    float3 neighborCoord = hexCoord + neighborOffset;
    float3 offsetNeighbor = offsetCoord(neighborCoord, agitation);
    float3 unskewedNeighbor = unskew(offsetNeighbor, hexSize);
    float dist = hexDistance(p, unskewedNeighbor, hexSize);
    float minDist2 = GLSL_min(minDists.y, GLSL_max(minDists.x, dist));
    float minDist1 = GLSL_min(minDists.x, dist);
    return (float2)(minDist1, minDist2);
}

float2 minDistancesFromPoint(float3 p, float hexSize, float agitation) {
    float3 centreCoord = hexCoord(p, hexSize);
    float2 minDists = (float2)(2.0f, 2.0f);
    float minXY = -2.0f;
    float maxXY = 2.0f;
    float z = 0.0f;
    for (float x = minXY; x <= maxXY; ++x) {
        for (float y = minXY; y <= maxXY; ++y) {
            if (GLSL_abs(x + y) <= 2.0f) {
                minDists = minDistances(p, centreCoord, (float3)(x, y, z), minDists, hexSize, agitation);
            }
        }
    }
    for (float z = -1.0f; z <= 1.0f; z += 2.0f) {
        float signZ = GLSL_sign(z);
        float halfSignZ = signZ / 2.0f;
        minXY = -1.5f - halfSignZ;
        maxXY = 1.5f - halfSignZ;
        for (float x = minXY; x <= maxXY; ++x) {
            for (float y = minXY; y <= maxXY; ++y) {
                if (!((x == signZ * 1.0f && y == signZ * 1.0f) || (x == signZ * -2.0f && y == signZ * -1.0f) || (x == signZ * -1.0f && y == signZ * -2.0f) || (x == signZ * -2.0f && y == signZ * -2.0f))) {
                    minDists = minDistances(p, centreCoord, (float3)(x, y, z), minDists, hexSize, agitation);
                }
            }
        }
    }
    for (float z = -2.0f; z <= 2.0f; z += 4.0f) {
        float signZ = GLSL_sign(z);
        minXY = -1.0f - signZ;
        maxXY = 1.0f - signZ;
        for (float x = minXY; x <= maxXY; ++x) {
            for (float y = minXY; y <= maxXY; ++y) {
                if (GLSL_abs(x + y) <= 2.0f) {
                    minDists = minDistances(p, centreCoord, (float3)(x, y, z), minDists, hexSize, agitation);
                }
            }
        }
    }
    return minDists;
}

