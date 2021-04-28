//=============================================================================================
// Collection of programs from lecture slides.
// Framework for assignments. Valid from 2020.
//
// Do not change it if you want to submit a homework.
// In the homework, file operations other than printf are prohibited.
//=============================================================================================
#define _USE_MATH_DEFINES		// M_PI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

// Resolution of screen
const unsigned int windowWidth = 600, windowHeight = 600;

//--------------------------
struct vec2 {
//--------------------------
	float x, y;

	vec2(float x0 = 0, float y0 = 0) { x = x0; y = y0; }
	vec2 operator*(float a) const { return vec2(x * a, y * a); }
	vec2 operator/(float a) const { return vec2(x / a, y / a); }
	vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }
	vec2 operator-() const { return vec2(-x, -y); }
};

inline float dot(const vec2& v1, const vec2& v2) {
	return (v1.x * v2.x + v1.y * v2.y);
}

inline float length(const vec2& v) { return sqrtf(dot(v, v)); }

inline vec2 normalize(const vec2& v) { return v * (1 / length(v)); }

inline vec2 operator*(float a, const vec2& v) { return vec2(v.x * a, v.y * a); }

//--------------------------
struct vec3 {
//--------------------------
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
	vec3(vec2 v) { x = v.x; y = v.y; z = 0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator-()  const { return vec3(-x, -y, -z); }
};

inline float dot(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

inline float length(const vec3& v) { return sqrtf(dot(v, v)); }

inline vec3 normalize(const vec3& v) { return v * (1 / length(v)); }

inline vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

inline vec3 operator*(float a, const vec3& v) { return vec3(v.x * a, v.y * a, v.z * a); }

//--------------------------
struct vec4 {
//--------------------------
	float x, y, z, w;

	vec4(float x0 = 0, float y0 = 0, float z0 = 0, float w0 = 0) { x = x0; y = y0; z = z0; w = w0; }
	float& operator[](int j) { return *(&x + j); }
	float operator[](int j) const { return *(&x + j); }

	vec4 operator*(float a) const { return vec4(x * a, y * a, z * a, w * a); }
	vec4 operator/(float d) const { return vec4(x / d, y / d, z / d, w / d); }
	vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
	vec4 operator-(const vec4& v)  const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
	vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
	void operator+=(const vec4 right) { x += right.x; y += right.y; z += right.z; w += right.w; }
};

inline float dot(const vec4& v1, const vec4& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
}

inline vec4 operator*(float a, const vec4& v) {
	return vec4(v.x * a, v.y * a, v.z * a, v.w * a);
}

//---------------------------
struct mat4 { // row-major matrix 4x4
//---------------------------
	vec4 rows[4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		rows[0][0] = m00; rows[0][1] = m01; rows[0][2] = m02; rows[0][3] = m03;
		rows[1][0] = m10; rows[1][1] = m11; rows[1][2] = m12; rows[1][3] = m13;
		rows[2][0] = m20; rows[2][1] = m21; rows[2][2] = m22; rows[2][3] = m23;
		rows[3][0] = m30; rows[3][1] = m31; rows[3][2] = m32; rows[3][3] = m33;
	}
	mat4(vec4 it, vec4 jt, vec4 kt, vec4 ot) {
		rows[0] = it; rows[1] = jt; rows[2] = kt; rows[3] = ot;
	}

	vec4& operator[](int i) { return rows[i]; }
	vec4 operator[](int i) const { return rows[i]; }
	operator float*() const { return (float*)this; }
};

inline vec4 operator*(const vec4& v, const mat4& mat) {
	return v[0] * mat[0] + v[1] * mat[1] + v[2] * mat[2] + v[3] * mat[3];
}

inline mat4 operator*(const mat4& left, const mat4& right) {
	mat4 result;
	for (int i = 0; i < 4; i++) result.rows[i] = left.rows[i] * right;
	return result;
}

inline mat4 TranslateMatrix(vec3 t) {
	return mat4(vec4(1,   0,   0,   0),
			    vec4(0,   1,   0,   0),
				vec4(0,   0,   1,   0),
				vec4(t.x, t.y, t.z, 1));
}

inline mat4 ScaleMatrix(vec3 s) {
	return mat4(vec4(s.x, 0,   0,   0),
			    vec4(0,   s.y, 0,   0),
				vec4(0,   0,   s.z, 0),
				vec4(0,   0,   0,   1));
}

inline mat4 RotationMatrix(float angle, vec3 w) {
	float c = cosf(angle), s = sinf(angle);
	w = normalize(w);
	return mat4(vec4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s, w.x*w.z*(1 - c) - w.y*s, 0),
			    vec4(w.x*w.y*(1 - c) - w.z*s, c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s, 0),
			    vec4(w.x*w.z*(1 - c) + w.y*s, w.y*w.z*(1 - c) - w.x*s, c * (1 - w.z*w.z) + w.z*w.z, 0),
			    vec4(0, 0, 0, 1));
}

//---------------------------
class Texture {
//---------------------------
	std::vector<vec4> load(std::string pathname, bool transparent, int& width, int& height) {
		FILE * file = fopen(pathname.c_str(), "r");
		if (!file) {
			printf("%s does not exist\n", pathname.c_str());
			width = height = 0;
			return std::vector<vec4>();
		}
		unsigned short bitmapFileHeader[27];					// bitmap header
		fread(&bitmapFileHeader, 27, 2, file);
		if (bitmapFileHeader[0] != 0x4D42) printf("Not bmp file\n");
		if (bitmapFileHeader[14] != 24) printf("Only true color bmp files are supported\n");
		width = bitmapFileHeader[9];
		height = bitmapFileHeader[11];
		unsigned int size = (unsigned long)bitmapFileHeader[17] + (unsigned long)bitmapFileHeader[18] * 65536;
		fseek(file, 54, SEEK_SET);
		std::vector<unsigned char> bImage(size);
		fread(&bImage[0], 1, size, file); 	// read the pixels
		fclose(file);
		std::vector<vec4> image(width * height);
		int i = 0;
		for (unsigned int idx = 0; idx < size; idx += 3) { // Swap R and B since in BMP, the order is BGR
			float alpha = (transparent) ? (bImage[idx] + bImage[idx + 1] + bImage[idx + 2]) / 3.0f / 256.0f : 1.0f;
			image[i++] = vec4(bImage[idx + 2] / 256.0f, bImage[idx + 1] / 256.0f, bImage[idx] / 256.0f, alpha);
		}
		return image;
	}

public:
	unsigned int textureId = 0;

	Texture() { textureId = 0; }

	Texture(std::string pathname, bool transparent = false) {
		textureId = 0;
		create(pathname, transparent);
	}

	Texture(int width, int height, const std::vector<vec4>& image, int sampling = GL_LINEAR) {
		textureId = 0;
		create(width, height, image, sampling);
	}

	Texture(const Texture& texture) {
		printf("\nError: Texture resource is not copied on GPU!!!\n");
	}

	void operator=(const Texture& texture) {
		printf("\nError: Texture resource is not copied on GPU!!!\n");
	}

	void create(std::string pathname, bool transparent = false) {
		int width, height;
		std::vector<vec4> image = load(pathname, transparent, width, height);
		if (image.size() > 0) create(width, height, image);
	}

	void create(int width, int height, const std::vector<vec4>& image, int sampling = GL_LINEAR) {
		if (textureId == 0) glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, &image[0]); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sampling); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sampling);
	}

	~Texture() {
		if (textureId > 0) glDeleteTextures(1, &textureId);
	}
};

//---------------------------
class GPUProgram {
//--------------------------
	unsigned int shaderProgramId = 0;
	unsigned int vertexShader = 0, geometryShader = 0, fragmentShader = 0;
	bool waitError = true;

	void getErrorInfo(unsigned int handle) { // shader error report
		int logLen, written;
		glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0) {
			std::string log(logLen, '\0');
			glGetShaderInfoLog(handle, logLen, &written, &log[0]);
			printf("Shader log:\n%s", log.c_str());
			if (waitError) getchar();
		}
	}

	bool checkShader(unsigned int shader, std::string message) { // check if shader could be compiled
		int OK;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
		if (!OK) {
			printf("%s!\n", message.c_str());
			getErrorInfo(shader);
			return false;
		}
		return true;
	}

	bool checkLinking(unsigned int program) { 	// check if shader could be linked
		int OK;
		glGetProgramiv(program, GL_LINK_STATUS, &OK);
		if (!OK) {
			printf("Failed to link shader program!\n");
			getErrorInfo(program);
			return false;
		}
		return true;
	}

	int getLocation(const std::string& name) {	// get the address of a GPU uniform variable
		int location = glGetUniformLocation(shaderProgramId, name.c_str());
		if (location < 0) printf("uniform %s cannot be set\n", name.c_str());
		return location;
	}

public:
	GPUProgram(bool _waitError = true) { shaderProgramId = 0; waitError = _waitError; }

	GPUProgram(const GPUProgram& program) {
		if (program.shaderProgramId > 0) printf("\nError: GPU program is not copied on GPU!!!\n");
	}

	void operator=(const GPUProgram& program) {
		if (program.shaderProgramId > 0) printf("\nError: GPU program is not copied on GPU!!!\n");
	}

	unsigned int getId() { return shaderProgramId; }

	bool create(const char * const vertexShaderSource,
		        const char * const fragmentShaderSource, const char * const fragmentShaderOutputName,
		        const char * const geometryShaderSource = nullptr)
	{
		// Create vertex shader from string
		if (vertexShader == 0) vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) {
			printf("Error in vertex shader creation\n");
			exit(1);
		}
		glShaderSource(vertexShader, 1, (const GLchar**)&vertexShaderSource, NULL);
		glCompileShader(vertexShader);
		if (!checkShader(vertexShader, "Vertex shader error")) return false;

		// Create geometry shader from string if given
		if (geometryShaderSource != nullptr) {
			if (geometryShader == 0) geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
			if (!geometryShader) {
				printf("Error in geometry shader creation\n");
				exit(1);
			}
			glShaderSource(geometryShader, 1, (const GLchar**)&geometryShaderSource, NULL);
			glCompileShader(geometryShader);
			if (!checkShader(geometryShader, "Geometry shader error")) return false;
		}

		// Create fragment shader from string
		if (fragmentShader == 0) fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) {
			printf("Error in fragment shader creation\n");
			exit(1);
		}

		glShaderSource(fragmentShader, 1, (const GLchar**)&fragmentShaderSource, NULL);
		glCompileShader(fragmentShader);
		if (!checkShader(fragmentShader, "Fragment shader error")) return false;

		shaderProgramId = glCreateProgram();
		if (!shaderProgramId) {
			printf("Error in shader program creation\n");
			exit(1);
		}
		glAttachShader(shaderProgramId, vertexShader);
		glAttachShader(shaderProgramId, fragmentShader);
		if (geometryShader > 0) glAttachShader(shaderProgramId, geometryShader);

		// Connect the fragmentColor to the frame buffer memory
		glBindFragDataLocation(shaderProgramId, 0, fragmentShaderOutputName);	// this output goes to the frame buffer memory

		// program packaging
		glLinkProgram(shaderProgramId);
		if (!checkLinking(shaderProgramId)) return false;

		// make this program run
		glUseProgram(shaderProgramId);
		return true;
	}

	void Use() { 		// make this program run
		glUseProgram(shaderProgramId);
	}

	void setUniform(int i, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform1i(location, i);
	}

	void setUniform(float f, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform1f(location, f);
	}

	void setUniform(const vec2& v, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform2fv(location, 1, &v.x);
	}

	void setUniform(const vec3& v, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform3fv(location, 1, &v.x);
	}

	void setUniform(const vec4& v, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniform4fv(location, 1, &v.x);
	}

	void setUniform(const mat4& mat, const std::string& name) {
		int location = getLocation(name);
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, mat);
	}

	void setUniform(const Texture& texture, const std::string& samplerName, unsigned int textureUnit = 0) {
		int location = getLocation(samplerName);
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, texture.textureId);
		}
	}

	~GPUProgram() { if (shaderProgramId > 0) glDeleteProgram(shaderProgramId); }
};
