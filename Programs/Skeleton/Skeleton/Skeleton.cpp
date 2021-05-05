﻿//=============================================================================================
// Harmadik hazifeladat Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : EWMK9A
// Neptun : Madi Istvan Laszlo
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
float dt = 0.0008;
bool spacePressed = false;
std::vector <vec2> tomegek;
float m = 0.009;
template<class T> struct Dnum {
    float f;
    T d;
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
	   return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
	   return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}
typedef Dnum<vec2> Dnum2;
const int tessellationLevel = 200;
struct Camera {
    //---------------------------
    vec3 wEye, wLookat, wVup;
public:
    Camera() {}
    mat4 V() {
	   vec3 w = normalize(wEye - wLookat);
	   vec3 u = normalize(cross(wVup, w));
	   vec3 v = cross(w, u);
	   return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
		  u.y, v.y, w.y, 0,
		  u.z, v.z, w.z, 0,
		  0, 0, 0, 1);
    }

    mat4 P() { return mat4(); };
};
struct PersCamera :Camera {
    float fov, asp, fp, bp;
public:
    PersCamera() {
	   asp = (float)windowWidth / windowHeight;
	   fov = 75.0f * (float)M_PI / 180.0f;
	   fp = 1; bp = 0.005;

    }

    mat4 P() {
	   return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
		  0, 1 / tan(fov / 2), 0, 0,
		  0, 0, -(fp + bp) / (bp - fp), -1,
		  0, 0, -2 * fp * bp / (bp - fp), 0);
    }
};
struct OrtoCamera :Camera {
    float w, h, n, f;
    OrtoCamera() {
	   w = 4;
	   h = 4;
	   n = 0;
	   f = 50;
    }
    mat4 P() {
	   return mat4(
		  2 / w, 0, 0, 0,
		  0, 2 / h, 0, 0,
		  0, 0, -2 / (f - n), 0,
		  0, 0, 0, 1
	   );
    }
};
struct Material {
    //---------------------------
    vec3 kd, ks, ka;
    float shininess;
};
struct Light {
    //---------------------------
    vec3 La, Le;
    vec4 wLightPos;
    vec4 srtartPos;

    float length(const vec4& v) { return sqrtf(dot(v, v)); }
    vec4 qszorzas(vec4 q1, vec4 q2) {
	   vec4 q;
	   q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
	   q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
	   q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
	   q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);

	   return q;
    }

    void Animate(float tstart, float tend, vec4 pivot) {
	   float dt = tend;

	   vec4 q = vec4(cosf(dt / 4), sinf(dt / 4) * cosf(dt) / 2, sinf(dt / 4) * sinf(dt) / 2, sinf(dt / 4) * sqrtf(3 / 4));
	   q = q / length(q);

	   vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
	   qinv = qinv / length(qinv);

	   wLightPos = qszorzas(q, srtartPos - pivot);
	   wLightPos = qszorzas(wLightPos, qinv);
	   wLightPos = wLightPos + pivot;
    }
};
class CheckerBoardTexture : public Texture {
public:
    CheckerBoardTexture(const int width, const int height) : Texture() {
	   std::vector<vec4> image(width * height);
	   const vec4 blue(0, 0.6, 0, 1);
	   for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
		  image[y * width + x] = (x & 1) ^ (y & 1) ? blue : blue;
	   }
	   create(width, height, image, GL_NEAREST);
    }
};
class RanadomColorTexture : public Texture {
public:
    RanadomColorTexture(const int width, const int height) : Texture() {
	   std::vector<vec4> image(width * height);
	   float x = ((double)rand() / (RAND_MAX));
	   const vec4 color(1, 0, 0, 1);
	   for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
		  image[y * width + x] = color;
	   }
	   create(width, height, image, GL_NEAREST);
    }
};
struct RenderState {
    mat4	           MVP, M, Minv, V, P;
    Material* material;
    std::vector<Light> lights;
    Texture* texture;
    vec3	           wEye;
};
class Shader : public GPUProgram {
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
	   setUniform(material.kd, name + ".kd");
	   setUniform(material.ks, name + ".ks");
	   setUniform(material.ka, name + ".ka");
	   setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
	   setUniform(light.La, name + ".La");
	   setUniform(light.Le, name + ".Le");
	   setUniform(light.wLightPos, name + ".wLightPos");
    }
};
class PhongShader : public Shader {
    const char* vertexSource = R"(
		#version 330
		precision highp float;
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye
		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 
		layout(location = 2) in vec2  vtxUV;
		out vec3 wNormal;		    
		out vec3 wView;             
		out vec3 wLight[8];		   
		out vec2 texcoord;
		out float melyseg;
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
	   int tmp =int(vtxPos.y);
			melyseg=vtxPos.y-tmp;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";
    const char* fragmentSource = R"(
    #version 330
    precision highp float;
    
    struct Light {
    vec3 La, Le;
    vec4 wLightPos;
    };
    
    struct Material {
    vec3 kd, ks, ka;
    float shininess;
    };
    
    uniform Material material;
    uniform Light[8] lights;    // light sources
    uniform int   nLights;
    uniform sampler2D diffuseTexture;
    
    in  vec3 wNormal;       // interpolated world sp normal
    in  vec3 wView;         // interpolated world sp view
    in  vec3 wLight[8];     // interpolated world sp illum dir
    in  vec2 texcoord;
    in  float melyseg;
    out vec4 fragmentColor; // output goes to frame buffer
    
    void main() {
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein
    vec3 texColor = texture(diffuseTexture, texcoord).rgb;
 vec3 ka;
    
    ka = material.ka * texColor/50/abs(floor(50*melyseg));
    if(melyseg==0) ka = material.ka * texColor*0.3;
    vec3 kd = material.kd * texColor;
    
    vec3 radiance = vec3(0, 0, 0);
    for(int i = 0; i < nLights; i++) {
    vec3 L = normalize(wLight[i]);
    vec3 H = normalize(L + V);
    float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
    // kd and ka are modulated by the texture
    float d = length(lights[i].wLightPos);
    radiance += ka * lights[i].La +
    (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(d, 2);
    }
    fragmentColor = vec4(radiance, 1);
    }
    )";

public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
	   Use();
	   setUniform(state.MVP, "MVP");
	   setUniform(state.M, "M");
	   setUniform(state.Minv, "Minv");
	   setUniform(state.wEye, "wEye");

	   setUniform(*state.texture, std::string("diffuseTexture"));
	   setUniformMaterial(*state.material, "material");

	   setUniform((int)state.lights.size(), "nLights");
	   for (unsigned int i = 0; i < state.lights.size(); i++) {
		  setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
	   }
    }
};
class PhongShader2 : public Shader {
    const char* vertexSource = R"(
		#version 330
		precision highp float;
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;
		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;
		out float melyseg;
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
	   int tmp =int(vtxPos.y);
			melyseg=vtxPos.y-tmp;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

    const char* fragmentSource = R"(
    #version 330
    precision highp float;
    
    struct Light {
    vec3 La, Le;
    vec4 wLightPos;
    };
    
    struct Material {
    vec3 kd, ks, ka;
    float shininess;
    };
    
    uniform Material material;
    uniform Light[8] lights;    
    uniform int   nLights;
    uniform sampler2D diffuseTexture;
    
    in  vec3 wNormal;       
    in  vec3 wView;         
    in  vec3 wLight[8];     
    in  vec2 texcoord;
    in  float melyseg;
    out vec4 fragmentColor; // output goes to frame buffer
    
    void main() {
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    if (dot(N, V) < 0) N = -N;   
    vec3 texColor = texture(diffuseTexture, texcoord).rgb;
 vec3 ka;
    if(floor(melyseg)==0)ka=vec3(0.1,0.1,0.1);
     ka = material.ka * texColor*0.34;
    vec3 kd = material.kd * texColor;
    
    vec3 radiance = vec3(0, 0, 0);
    for(int i = 0; i < nLights; i++) {
    vec3 L = normalize(wLight[i]);
    vec3 H = normalize(L + V);
    float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
    // kd and ka are modulated by the texture
    float d = length(lights[i].wLightPos);
    radiance += ka * lights[i].La +
    (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(d, 2);
    }
    fragmentColor = vec4(radiance, 1);
    }
    )";

public:
    PhongShader2() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
	   Use();
	   setUniform(state.MVP, "MVP");
	   setUniform(state.M, "M");
	   setUniform(state.Minv, "Minv");
	   setUniform(state.wEye, "wEye");

	   setUniform(*state.texture, std::string("diffuseTexture"));
	   setUniformMaterial(*state.material, "material");

	   setUniform((int)state.lights.size(), "nLights");
	   for (unsigned int i = 0; i < state.lights.size(); i++) {
		  setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
	   }
    }
};
class Geometry {
protected:
    unsigned int vao, vbo;
public:
    Geometry() {
	   glGenVertexArrays(1, &vao);
	   glBindVertexArray(vao);
	   glGenBuffers(1, &vbo);
	   glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
	   glDeleteBuffers(1, &vbo);
	   glDeleteVertexArrays(1, &vao);
    }
};
struct VertexData {
    vec3 position, normal;
    vec2 texcoord;
};
class ParamSurface : public Geometry {
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

    VertexData GenVertexData(float u, float v) {
	   VertexData vtxData;
	   vtxData.texcoord = vec2(u, v);
	   Dnum2 X, Y, Z;
	   Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
	   eval(U, V, X, Y, Z);
	   vtxData.position = vec3(X.f, Y.f, Z.f);
	   vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
	   vtxData.normal = cross(drdU, drdV);
	   return vtxData;
    }

    void create(int N = tessellationLevel, int M = tessellationLevel) {
	   nVtxPerStrip = (M + 1) * 2;
	   nStrips = N;
	   std::vector<VertexData> vtxData;
	   for (int i = 0; i < N; i++) {
		  for (int j = 0; j <= M; j++) {
			 vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
			 vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
		  }
	   }
	   glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
	   glEnableVertexAttribArray(0);
	   glEnableVertexAttribArray(1);
	   glEnableVertexAttribArray(2);
	   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
	   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	   glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }

    void Draw() {
	   glBindVertexArray(vao);
	   for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
    }
};
class Sphere : public ParamSurface {
    //---------------------------
public:
    Sphere() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
	   U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
	   X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
    }
};
float distance(vec2 v1, vec2 v2) {
    return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y));
}
class Sheet : public ParamSurface {
    Dnum2 d = Dnum2();
public:
    Sheet() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {

	   U = U * 2 - 1;
	   V = V * 2 - 1;
	   X = U * 2;
	   Z = V * 2;
	   Y = 0;


	   int x = 0;
	   for (vec2 v : tomegek)
	   {
		  x++;
		  Dnum2 tomegpontX = Dnum2(v.x);
		  Dnum2 tomegpontY = Dnum2(v.y);

		  Y = Y + Pow(Pow(Pow(X - tomegpontX, 2) + Pow(Z - tomegpontY, 2), 0.500) + 0.005 * 4, -1) * -1 * m * x;
	   }

    }


};

struct Object {
    //---------------------------
    Shader* shader;
    Material* material;
    Texture* texture;
    Geometry* geometry;
    vec3 scale, translation, rotationAxis, v, a;
    float rotationAngle;

public:
    bool rajzoljae = true;
    bool pov = false;
    Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
	   scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
	   shader = _shader;
	   texture = _texture;
	   material = _material;
	   geometry = _geometry;
	   v = vec3(0, 0, 0);
	   a = vec3(0, 0, 0);
    }

    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
	   M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
	   Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }
    void Draw(RenderState state) {
	   if (!rajzoljae)return;
	   mat4 M, Minv;
	   SetModelingTransform(M, Minv);
	   state.M = M;
	   state.Minv = Minv;
	   state.MVP = state.M * state.V * state.P;
	   state.material = material;
	   state.texture = texture;
	   shader->Bind(state);
	   geometry->Draw();
    }
    void setV(vec3 _v) {
	   v = _v;
    }
    virtual void Animate(float tstart, float tend, Camera& camera) {
	   int tmp1 = 2, tmp2 = 4;
	   if (translation.z > tmp1)
	   {
		  translation.z -= tmp2;
	   }
	   if (translation.x > tmp1)
	   {
		  translation.x -= tmp2;
	   }
	   if (-translation.z > tmp1)
	   {
		  translation.z += tmp2;

	   }
	   if (-translation.x > tmp1)
	   {
		  translation.x += tmp2;

	   }
	   Dnum2 U = Dnum2(translation.x, vec2(1, 0)), V = Dnum2(translation.z, vec2(0, 1));

	   Dnum2 Y = 0, X = U, Z = V;

	   Dnum2 tomegpontX;
	   Dnum2 tomegpontY;
	   int x = 1;
	   vec3 norm = vec3(0, 1, 0);
	   if (abs(v.x) > 0.1 || abs(v.y) > 0.1 || abs(v.z) > 0.1)
	   {

		  for (vec2 v : tomegek)
		  {
			 tomegpontX = Dnum2(v.x);
			 tomegpontY = Dnum2(v.y);
			 Y = Y + Pow(Pow(Pow(X - tomegpontX, 2) + Pow(Z - tomegpontY, 2), 0.500) + 1 / 200 * 4, -1) * -1 * m * x;
			 x++;
		  }
		  vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		  norm = cross(drdU, drdV);


		  if (norm.y > 0)
		  {
			 a = 5000 * norm;
		  }
		  else
		  {
			 a = -5000 * norm;
		  }
		  a = a * vec3(1, 0, 1);
		  translation.y = Y.f + 0.05 * -norm.y;
		  v = v + dt * a;
		  for (vec2 tomeg : tomegek) {
			 if (distance(vec2(translation.x, translation.z), tomeg) < 0.07)
			 {
				rajzoljae = false;
				pov = false;

			 }
		  }

	   }
	   translation = translation + v * dt;
	   if (pov)
	   {
		  if (spacePressed)
		  {
			 rajzoljae = false;
			 vec3 tmp = v;
			 if (0 == v.x && 0 == v.y && 0 == v.z)
			 {
				tmp = vec3(1, 0, 1);
			 }
			 camera.wEye = this->translation;
			 camera.wLookat = this->translation + tmp;
			 camera.wVup = vec3(0, 1, 0);
		  }
		  else {
			 rajzoljae = true;

			 camera.wEye = vec3(0, 1, 0);
			 camera.wLookat = vec3(0, 0, 0);
			 camera.wVup = vec3(1, 0, 0);
		  }
	   }
    }
};
std::vector<Object*> objects;

class Scene {
    OrtoCamera camera = OrtoCamera();
    PersCamera pCamera;
    vec3 eye = vec3(0, 1, 0);
    std::vector<Light> lights;
public:
    void createSheet() {
	   Shader* phongShader = new PhongShader();

	   Material* material0 = new Material;
	   material0->kd = vec3(30, 30, 30);
	   material0->ks = vec3(30, 30, 30);
	   material0->ka = vec3(11, 11, 11);
	   material0->shininess = 10;
	   Texture* texture15x20 = new CheckerBoardTexture(15, 20);
	   Geometry* sheet = new Sheet();
	   Object* sheetObject = new Object(phongShader, material0, texture15x20, sheet);
	   sheetObject->translation = vec3(0, 0, 0);
	   sheetObject->scale = vec3(1, 1, 1);
	   sheetObject->rotationAxis = vec3(0, 1, 0);
	   objects[0] = sheetObject;

    }
    void Build() {
	   Shader* phongShader = new PhongShader();
	   Shader* phongShader2 = new PhongShader2();
	   Material* material0 = new Material;
	   material0->kd = vec3(1, 1, 1);
	   material0->ks = vec3(2, 3, 1);
	   material0->ka = vec3(1, 1, 1);
	   material0->shininess = 10;



	   Texture* rndttexture = new RanadomColorTexture(15, 20);
	   Texture* texture15x20 = new CheckerBoardTexture(15, 20);

	   Geometry* sphere = new Sphere();
	   Geometry* sheet = new Sheet();

	   Object* sphereObject1 = new Object(phongShader2, material0, rndttexture, sphere);
	   sphereObject1->translation = vec3(-1.85, 0.1, -1.85);
	   sphereObject1->scale = vec3(0.07f, 0.07f, 0.07f);
	   sphereObject1->pov = true;
	   Object* sheetObject = new Object(phongShader, material0, texture15x20, sheet);
	   sheetObject->translation = vec3(0, 0, 0);
	   sheetObject->scale = vec3(1, 1, 1);
	   sheetObject->rotationAxis = vec3(0, 1, 0);
	   objects.push_back(sheetObject);
	   objects.push_back(sphereObject1);

	   camera.wEye = eye;
	   camera.wLookat = vec3(0, 0, 0);
	   camera.wVup = vec3(1, 0, 0);

	   lights.resize(2);
	   lights[0].wLightPos = vec4(5, 5, 4, 0);
	   lights[0].La = vec3(1, 1, 1);
	   lights[0].Le = vec3(1, 1, 1);
	   lights[0].srtartPos = lights[0].wLightPos;

	   lights[1].wLightPos = vec4(5, 10, 20, 0);
	   lights[1].La = vec3(1, 2, 1);
	   lights[1].Le = vec3(1, 1, 1);
	   lights[1].srtartPos = lights[1].wLightPos;



    }

    void Render() {
	   RenderState state;
	   if (spacePressed) {
		  state.wEye = pCamera.wEye;
		  state.V = pCamera.V();
		  state.P = pCamera.P();
	   }
	   else
	   {
		  state.wEye = camera.wEye;
		  state.V = camera.V();
		  state.P = camera.P();
	   }

	   state.lights = lights;
	   for (Object* obj : objects) obj->Draw(state);
	   objects[0]->Draw(state);
    }

    void Animate(float tstart, float tend) {


	   lights[0].Animate(tstart, tend, lights[1].srtartPos);
	   lights[1].Animate(tstart, tend, lights[0].srtartPos);
	   int x = 0;
	   for (Object* obj : objects) {
		  if (obj->pov)
		  {
			 x++;
		  }
	   }
	   if (x == 0) {
		  for (Object* obj : objects) {
			 if (obj->rajzoljae && !obj->pov)
			 {
				obj->pov = true;
			 }
		  }
	   }
	   for (Object* obj : objects) {
		  obj->Animate(tstart, tend, pCamera);

	   }
    }


    void startSphere(vec2 velocity) {
	   objects[objects.size() - 1]->setV(vec3(velocity.y, 0, velocity.x) / 100);
	   Shader* phongShader2 = new PhongShader2();


	   Material* material0 = new Material;
	   material0->kd = vec3(1, 1, 1);
	   material0->ks = vec3(2, 3, 1);
	   material0->ka = vec3(1, 1, 1);
	   material0->shininess = 10;

	   Texture* rndttexture = new RanadomColorTexture(15, 20);
	   Geometry* sphere = new Sphere();
	   Object* sphereObject1 = new Object(phongShader2, material0, rndttexture, sphere);
	   sphereObject1->translation = vec3(-1.85, 0.1, -1.85);
	   sphereObject1->scale = vec3(0.07f, 0.07f, 0.07f);
	   objects.push_back(sphereObject1);


    }
};
Scene scene;
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.Render();
    glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == ' ')
    {
	   spacePressed = !spacePressed;
	   scene.createSheet();

    }
}
void onKeyboardUp(unsigned char key, int pX, int pY) { }
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && !spacePressed)
    {
	   vec2 pressedAt = vec2(-pX, pY);
	   vec2 v = vec2(0, 600) - pressedAt;
	   scene.startSphere(v);
    }
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN && !spacePressed)
    {
	   float cX = 2.0f * pX / windowWidth - 1;
	   float cY = 1.0f - 2.0f * pY / windowHeight;
	   tomegek.push_back(vec2(2 * cY, 2 * cX));
	   scene.createSheet();
    }


}
void onMouseMotion(int pX, int pY) {
}
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f;
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
	   float Dt = fmin(dt, tend - t);
	   scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}