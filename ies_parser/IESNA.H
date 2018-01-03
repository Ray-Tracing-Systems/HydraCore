/*
 *************************************************************************
 *
 *  IESNA.H - IESNA LM-63 Photometric Data Module Include File
 *
 *  Version:    1.00D
 *
 *  History:    95/08/15 - Created.
 *              95/08/29 - Version 1.00A release.
 *              95/09/01 - Revised IE_MaxLabel and IE_MaxLine definitions.
 *              95/09/04 - Version 1.00B release.
 *              96/01/28 - Added CIE luminaire type classification
 *                         definitions.
 *                       - Added calculated photometric data definitions.
 *                       - Added IE_Calc and IE_Zonal data structures.
 *                       - Added IE_CalcData and IE_CalcCU_Array function
 *                         prototypes.
 *                       - Added coefficients of utilization array
 *                         dimensions.
 *                       - Added IE_CU_Array and IE_CIE_Type
 *                         declarations.
 *                       - Added definition for pi.
 *              96/01/30 - Version 1.00C release.
 *              98/03/09 - Version 1.00D release.
 *
 *  Compilers:  Any ANSI C-compliant compiler
 *
 *  Author:     Ian Ashdown, P. Eng.
 *              byHeart Consultants Limited
 *              620 Ballantree Road
 *              West Vancouver, B.C.
 *              Canada V7S 1W3
 *              Tel. (604) 922-6148
 *              Fax. (604) 987-7621
 *
 *  Copyright 1995-1998 byHeart Consultants Limited
 *
 *  Permission: The following source code is copyrighted. However, it may
 *              be freely copied, redistributed, and modified for personal
 *              use or for royalty-free inclusion in commercial programs.
 *
 *************************************************************************
 */

#ifndef _IESNA_H
#define _IESNA_H

#include <stdio.h>

#define FALSE   0
#define TRUE    1

#define PI      3.141592654

#define IE_MaxLabel     80      /* Maximum label line width             */
#define IE_MaxLine      130     /* Maximum non-label line width         */

/* Calculated photometric data                                          */
#define IE_INDEX_NONE   -1      /* Invalid candela array index          */
#define IE_HORZ         9       /* Number of horizontal angles (0-180)  */
#define IE_HORZ_90      IE_HORZ/2       /* 90-degree horizontal angle   */
#define IE_VERT_CAND    37      /* Number of vertical angles (candela)  */
#define IE_VERT_90      IE_VERT_CAND/2  /* 90-degree vertical angle     */
#define IE_VERT_180     IE_VERT_CAND-1  /* 180-degree vertical angle    */
#define IE_VERT_FLUX    18      /* Number of vertical angles (flux)     */
#define IE_V_ANGLE      5.0     /* Vertical angle increment             */
#define IE_H_ANGLE      22.5    /* Horizontal angle increment           */
#define IE_ZONES        9       /* Number of zones                      */

/* CIE luminaire type classification                                    */
#define IE_CIE_1        0       /* Direct                               */
#define IE_CIE_2        1       /* Semi-direct                          */
#define IE_CIE_3        2       /* General diffuse                      */
#define IE_CIE_4        3       /* Semi-indirect                        */
#define IE_CIE_5        4       /* Indirect                             */

/* Coefficients of utilization array dimensions                         */
#define IE_CU_ROWS      11      /* Room cavity ratios                   */
#define IE_CU_COLS      18      /* Ceiling/wall reflectances            */

typedef int BOOL;               /* Boolean flag                         */

enum IES_format                 /* Format                               */
    {
      IESNA_86,                 /* LM-63-1986                           */
      IESNA_91,                 /* LM-63-1991                           */
      IESNA_95                  /* LM-63-1995                           */
    };

enum  IES_LampVHT                   /* Lamp-to-luminaire geometry           */
      {
        LampVert = 1,           /* Lamp vertical base up or down        */
        LampHorz = 2,           /* Lamp horizontal                      */
        LampTilt = 3            /* Lamp tilted                          */
      };

enum IES_Units                  /* Measurement units                    */
  {
    Feet = 1,                   /* Imperial                             */
    Meters = 2                  /* Standard Internationale              */
  };

enum IES_Gonio_Type                       /* Photometric goniometer type          */
    {
      Type_A = 3,               /* Type A                               */
      Type_B = 2,               /* Type B                               */
      Type_C = 1                /* Type C                               */
    };

  struct IE_Label               /* Label line linked list element       */
  {
    char *line;                 /* Label line pointer                   */
    struct IE_Label *pnext;     /* Next list element pointer            */
  };


typedef struct IE_Data          /* IESNA Standard File data             */
{
  struct                        /* File information                     */
  {
    char *name;                 /* Name                                 */
    enum IES_format format;
  }
  file;

  struct IE_Label  *plline;     /* Label line linked list pointer       */

  struct                        /* Lamp data                            */
  {
    int num_lamps;              /* Number of lamps                      */
    float lumens_lamp;          /* Lumens per lamp                      */
    float multiplier;           /* Candela multiplying factor           */
    char *tilt_fname;           /* TILT file name pointer (optional)    */

    struct                      /* TILT data structure                  */
    {
      enum IES_LampVHT orientation;

      int num_pairs;            /* # of angle-multiplying factor pairs  */
      float *angles;            /* Angles array pointer                 */
      float *mult_factors;      /* Multiplying factors array pointer    */
    }
    tilt;
  }
  lamp;

  enum IES_Units units;

  struct                        /* Luminous cavity dimensions           */
  {
    float width;                /* Opening width                        */
    float length;               /* Opening length                       */
    float height;               /* Cavity height                        */
  }
  dim;

  struct                        /* Electrical data                      */
  {
    float ball_factor;          /* Ballast factor                       */
    float blp_factor;           /* Ballast-lamp photometric factor      */
    float input_watts;          /* Input watts                          */
  }
  elec;

  struct                        /* Photometric data                     */
  {
    enum IES_Gonio_Type gonio_type;

    int num_vert_angles;        /* Number of vertical angles            */
    int num_horz_angles;        /* Number of horizontal angles          */

    float *vert_angles;         /* Vertical angles array                */
    float *horz_angles;         /* Horizontal angles array              */

    float **pcandela;           /* Candela values array pointers array  */
  }
  photo;
}
IE_DATA;

typedef struct IE_Calc          /* Calculated photometric data          */
{
  /* NOTE: The candlepower distribution array is ordered as follows:    */
  /*                                                                    */
  /*   Horizontal: { 0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,  */
  /*                 180.0 }                                            */
  /*   Vertical:   { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,  */
  /*                 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115,    */
  /*                 120, 125, 130, 135, 140, 145, 150, 155, 160, 165,  */
  /*                 170, 175, 180 }                                    */

  long candela[IE_HORZ][IE_VERT_CAND];  /* Candlepower distribution     */

  int h_angle[IE_HORZ];         /* Valid horizontal angle index array   */
  int v_angle[IE_VERT_CAND];    /* Valid vertical angle index array     */

  int horz_num;         /* Number of horizontal angles                  */
  int vert_num;         /* Number of vertical angles                    */

  /* NOTE: The flux distribution arrays are ordered as follows:         */
  /*                                                                    */
  /*   { 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135,     */
  /*     145, 155, 165, 175 }                                           */

  long flux[IE_VERT_FLUX];      /* Lumen distribution                   */

  /* NOTE: The zonal lumens, percent lamp lumens and percent fixture    */
  /*       lumens arrays are ordered as follows:                        */
  /*                                                                    */
  /*   { 0-30, 0-40, 0-60, 0-90, 90-120, 90-130, 90-150, 90-180, 0-180  */

  long zonal_lm[IE_ZONES];      /* Zonal lumens                         */
  int lamp_pct[IE_ZONES];       /* Percent lamp lumens                  */
  int fixt_pct[IE_ZONES];       /* Percent fixture lumens               */

  double efficiency;            /* Luminaire efficiency                 */
  double total_lm;              /* Total lamp lumens                    */

  int cie_type;                 /* CIE luminaire type classification    */

  double cu;            /* Coefficient of utilization                   */
  double wec;           /* Wall luminous exitance coefficient           */
  double ccec;          /* Ceiling cavity luminous exitance coefficient */
  double wdrc;          /* Wall direct radiation coefficient            */

  /* Coefficients of Utilization array                                  */
  int IE_CU_Array[IE_CU_ROWS][IE_CU_COLS];
}
IE_CALC;

extern const char *IE_CIE_Type[5];

#ifdef WIN32

  extern BOOL IE_ReadFile( char *, IE_DATA * );
  extern BOOL IE_CalcData( IE_DATA *, IE_CALC * );
  extern void IE_Flush( IE_DATA * );

#else

  extern "C" BOOL IE_ReadFile( char *, IE_DATA * );
  extern "C" BOOL IE_CalcData( IE_DATA *, IE_CALC * );
  extern "C" void IE_Flush( IE_DATA * );

#endif

#endif

