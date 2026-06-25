/* Minimal raylib stub for unit testing — NOT the real raylib */
#pragma once
#include <stdarg.h>
#include <stdio.h>

typedef struct { unsigned char r, g, b, a; } Color;
typedef struct { float x, y, width, height; } Rectangle;
typedef struct { float x, y; } Vector2;
typedef struct { int id; } Texture2D;

#define KEY_ESCAPE 256
#define WHITE      (Color){255,255,255,255}
#define GRAY       (Color){130,130,130,255}
#define LIGHTGRAY  (Color){200,200,200,255}
#define BLACK      (Color){0,0,0,255}
#define RED        (Color){230,41,55,255}
#define BLANK      (Color){0,0,0,0}

static inline void InitWindow(int w, int h, const char* t){}
static inline void SetTargetFPS(int fps){}
static inline void CloseWindow(void){}
static inline int  IsWindowReady(void){return 0;}
static inline int  WindowShouldClose(void){return 0;}
static inline void BeginDrawing(void){}
static inline void EndDrawing(void){}
static inline void ClearBackground(Color c){}
static inline void DrawRectangle(int x,int y,int w,int h,Color c){}
static inline void DrawRectangleLines(int x,int y,int w,int h,Color c){}
static inline void DrawText(const char* t,int x,int y,int s,Color c){}
static inline int  IsKeyDown(int k){return 0;}
static inline int  IsKeyPressed(int k){return 0;}

static char _textbuf[256];
static inline const char* TextFormat(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(_textbuf, sizeof(_textbuf), fmt, args);
    va_end(args);
    return _textbuf;
}
