/*
 *************************************************************************
 *
 *  IES_READ.C - IESNA LM-63 Photometric Data Test Module
 *
 *  Version:    1.00D
 *
 *  History:    95/08/15 - Created.
 *              95/08/29 - Version 1.00A release.
 *              95/08/03 - Revised photmetric data display.
 *              95/09/04 - Version 1.00B release.
 *              96/01/29 - Added PhotoData and PhotoCalc global data
 *                         structures.
 *                       - Revised DisplayPhotoData function.
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

#include <stdio.h>
#include <string.h>
#include "IESNA.H"

static IE_DATA PhotoData;
static IE_CALC PhotoCalc;

static void DisplayPhotoData( BOOL );

/*
 *************************************************************************
 *
 *  main - Executive Function
 *
 *  Purpose:    Executive function to demonstrate use of IESNA LM-63
 *              Photometric Data Module functions.
 *
 *  Setup:      int main
 *              (
 *                int argc,
 *                char **argv;
 *              )
 *
 *  Usage:      ies_read <file>
 *
 *  Where:      file is the file name of an IESNA LM-63 standard photo-
 *                metric data file.
 *
 *************************************************************************
 */

int main( int argc, char **argv )
{
  BOOL status;

  /* Read IESNA photometric data file                                   */
  status = IE_ReadFile(argv[1], &PhotoData);

  if (status == TRUE)
  {
    /* Calculate additional photometric data                            */
    status = IE_CalcData(&PhotoData, &PhotoCalc);

    DisplayPhotoData(status);   /* Display photometric data             */
    IE_Flush(&PhotoData);       /* Release photometric data memory      */

    return 0;
  }
  else
    return 2;
}

/*
 *************************************************************************
 *
 *  DisplayPhotoData - Display Photometric Data
 *
 *  Purpose:    To display the photometric data
 *
 *  Setup:      static void DisplayPhotoData
 *              (
 *                BOOL calc_flag
 *              )
 *
 *  Where:      calc_flag is a Boolean flag which if TRUE indicates that
 *                the calculated photometric data is valid.
 *
 *************************************************************************
 */

static void DisplayPhotoData( BOOL calc_flag )
{
  struct IE_Label *plabel;
  int vert;
  int horz;
  int i, j, k;

  /* Display file information                                           */
  puts("              Ledalite IES Photometric Data File Utility");
  puts("              ------------------------------------------");
  puts("                            Version 1.00C\n");
  puts("Photometric Data File Information");
  puts("---------------------------------");
  printf("File name:       %s\n",
      PhotoData.file.name);
  printf("File format:     ");
  if (PhotoData.file.format == IESNA_86)
    printf("%s\n", "LM-63-1986");
  else if (PhotoData.file.format == IESNA_91)
    printf("%s\n", "LM-63-1991");
  else
    printf("%s\n", "LM-63-1995");
  printf("TILT file name = %s\n\n", PhotoData.lamp.tilt_fname);

  /* Display label lines                                                */
  puts("Luminaire Description");
  puts("---------------------");
  plabel = PhotoData.plline;
  while (plabel)
  {
    printf("%s\n", plabel->line);
    plabel = plabel->pnext;
  }
  putchar('\n');

  /* Display lamp data                                                  */
  puts("Lamp Data");
  puts("---------");
  printf("Number of lamps =            %-2d\n", PhotoData.lamp.num_lamps);
  printf("Lumens per lamp =     %8.2f\n", PhotoData.lamp.lumens_lamp);

  /* Display lamp tilt data (if any)                                    */
  if (strcmp(PhotoData.lamp.tilt_fname, "NONE") != 0)
  {
    switch (PhotoData.lamp.tilt.orientation)
   {
      case LampVert:
        puts("Lamp orientation =    Vertical");
        break;
      case LampHorz:
        puts("Lamp orientation =  Horizontal");
        break;
      case LampTilt:
        puts("Lamp orientation =      Tilted");
        break;
      default:
        puts("Lamp orientation =     Unknown");
        break;
    }

    printf("Number A-MF pairs =         %-3d\n",
        PhotoData.lamp.tilt.num_pairs);
    for (vert = 0; vert < PhotoData.lamp.tilt.num_pairs; vert++)
      printf("%8.2f - %8.2f\n", PhotoData.lamp.tilt.angles[vert],
          PhotoData.lamp.tilt.mult_factors[vert]);
  }

  /* Display luminaire dimensions                                       */
  puts("\nLuminaire Dimensions");
  puts("--------------------");
  printf("Measurement units =       ");
  if (PhotoData.units == Feet)
    puts("Feet");
  else
    puts("Meters");
  printf("Width =               %8.2f\n", PhotoData.dim.width);
  printf("Length =              %8.2f\n", PhotoData.dim.length);
  printf("Height =              %8.2f\n\n", PhotoData.dim.height);

  /* Display luminaire electrical data                                  */
  puts("Electrical Data");
  puts("---------------");
  printf("Ballast factor =      %8.2f\n", PhotoData.elec.ball_factor);

  /* Ballast-lamp photometric factor is not defined for LM-63-1995      */
  if ((PhotoData.file.format == IESNA_86) || (PhotoData.file.format == IESNA_91))
    printf("Ballast-lamp factor = %8.2f\n", PhotoData.elec.blp_factor);

  printf("Ballast watts =       %8.2f\n\n", PhotoData.elec.input_watts);

  /* Display photometric measurements                                   */
  puts("Photometric Data");
  puts("----------------");
  printf("Multiplier =          %8.2f\n", PhotoData.lamp.multiplier);
  printf("Goniometer type =       Type ");

  switch (PhotoData.photo.gonio_type)
  {
    case Type_A:
      puts("A");
      break;
    case Type_B:
      puts("B");
      break;
    case Type_C:
      puts("C");
      break;
    default:
      puts("Unknown");
      break;
  }

  if (calc_flag == FALSE)
  {
    printf("Vertical angles =            %-3d\n",
        PhotoData.photo.num_vert_angles);
    printf("Horizontal angles =          %-3d\n",
        PhotoData.photo.num_horz_angles);

    for (horz = 0; horz < PhotoData.photo.num_horz_angles; horz++)
    {
      printf("\nHorizontal Angle =    %8.2f\n",
          PhotoData.photo.horz_angles[horz]);
      for (vert = 0; vert < PhotoData.photo.num_vert_angles; vert++)
      {
        printf("  %8.2f - %8.2f cd\n",
            PhotoData.photo.vert_angles[vert],
            PhotoData.photo.pcandela[horz][vert]);
      }
    }
    puts("\nNOTE: This file does not contain sufficient photometric data"
        " to");
    puts("      accurately calculate the following information:\n");
    puts("      1. Luminaire efficiency");
    puts("      2. Zonal lumens");
    puts("      3. CIE distribution type");
    puts("      4. Coefficients of Utilization");
  }
  else
  {
    puts("\nCalculated Information");
    puts("----------------------");
    printf("Total Lamp Lumens:     %5.2f\n", PhotoCalc.total_lm);
    printf("Luminaire Efficiency:  %2.1f %%\n\n",
        PhotoCalc.efficiency);
    puts("                      CANDELA DISTRIBUTION                  "
        "FLUX");
    puts("                      --------------------                  "
        "----");
    puts("       0.0  22.5  45.0  67.5  90.0 112.5 135.0 157.5 180.0\n");
    printf("   0 %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld\n",
        PhotoCalc.candela[0][0], PhotoCalc.candela[1][0],
        PhotoCalc.candela[2][0], PhotoCalc.candela[3][0],
        PhotoCalc.candela[4][0], PhotoCalc.candela[5][0],
        PhotoCalc.candela[6][0], PhotoCalc.candela[7][0],
        PhotoCalc.candela[8][0]);

    for (i = 1, j = 5, k = 1; i <= 9; i++, j += 10, k += 2)
    {
      printf(" %3d %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld\n",
          j, PhotoCalc.candela[0][k], PhotoCalc.candela[1][k],
          PhotoCalc.candela[2][k], PhotoCalc.candela[3][k],
          PhotoCalc.candela[4][k], PhotoCalc.candela[5][k],
          PhotoCalc.candela[6][k], PhotoCalc.candela[7][k],
          PhotoCalc.candela[8][k], PhotoCalc.flux[i - 1]);
    }

    printf("  90 %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld\n",
        PhotoCalc.candela[0][18], PhotoCalc.candela[1][18],
        PhotoCalc.candela[2][18], PhotoCalc.candela[3][18],
        PhotoCalc.candela[4][18], PhotoCalc.candela[5][18],
        PhotoCalc.candela[6][18], PhotoCalc.candela[7][18],
        PhotoCalc.candela[8][18]);

    for (i = 1, j = 95, k = 19; i <= 9; i++, j += 10, k += 2)
    {
      printf(" %3d %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld\n",
          j, PhotoCalc.candela[0][k], PhotoCalc.candela[1][k],
          PhotoCalc.candela[2][k], PhotoCalc.candela[3][k],
          PhotoCalc.candela[4][k], PhotoCalc.candela[5][k],
          PhotoCalc.candela[6][k], PhotoCalc.candela[7][k],
          PhotoCalc.candela[8][k], PhotoCalc.flux[i + 8]);
    }

    printf(" 180 %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld %5ld\n\n",
        PhotoCalc.candela[0][36], PhotoCalc.candela[1][36],
        PhotoCalc.candela[2][36], PhotoCalc.candela[3][36],
        PhotoCalc.candela[4][36], PhotoCalc.candela[5][36],
        PhotoCalc.candela[6][36], PhotoCalc.candela[7][36],
        PhotoCalc.candela[8][36]);
    puts("           ZONAL LUMEN SUMMARY");
    puts("           -------------------");
    puts("   ZONE      LUMENS   %% LAMP   %% FIXT\n");
    printf("  0 - 90     %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[3], PhotoCalc.lamp_pct[3],
        PhotoCalc.fixt_pct[3]);
    printf("  0 - 30     %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[0], PhotoCalc.lamp_pct[0],
        PhotoCalc.fixt_pct[0]);
    printf("  0 - 40     %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[1], PhotoCalc.lamp_pct[1],
        PhotoCalc.fixt_pct[1]);
    printf("  0 - 60     %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[2], PhotoCalc.lamp_pct[2],
        PhotoCalc.fixt_pct[2]);
    printf(" 90 - 120    %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[4], PhotoCalc.lamp_pct[4],
        PhotoCalc.fixt_pct[4]);
    printf(" 90 - 130    %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[5], PhotoCalc.lamp_pct[5],
        PhotoCalc.fixt_pct[5]);
    printf(" 90 - 150    %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[6], PhotoCalc.lamp_pct[6],
        PhotoCalc.fixt_pct[6]);
    printf(" 90 - 180    %6ld     %4d     %4d\n",
        PhotoCalc.zonal_lm[7], PhotoCalc.lamp_pct[7],
        PhotoCalc.fixt_pct[7]);
    printf("  0 - 180    %6ld     %4d     %4d\n\n",
        PhotoCalc.zonal_lm[8], PhotoCalc.lamp_pct[8],
        PhotoCalc.fixt_pct[8]);
    printf("CIE Classification:  Type %s\n\n",
        IE_CIE_Type[PhotoCalc.cie_type]);
    puts("                    COEFFICIENTS OF UTILIZATION");
    puts("                    ---------------------------");
    puts(" RC       80            70           50         30         10 "
        "      0\n");
    puts(" RW   70 50 30 10   70 50 30 10   50 30 10   50 30 10   50 30 "
        "10    0");
    puts(" -------------------------------------------------------------"
        "-------\n");

    for (i = 0; i <= 10; i++)
    {
      printf(" %2d   %2d %2d %2d %2d   %2d %2d %2d %2d   %2d %2d %2d   "
          "%2d %2d %2d   %2d %2d %2d   %2d\n", i,
          PhotoCalc.IE_CU_Array[i][0], PhotoCalc.IE_CU_Array[i][1],
          PhotoCalc.IE_CU_Array[i][2], PhotoCalc.IE_CU_Array[i][3],
          PhotoCalc.IE_CU_Array[i][4], PhotoCalc.IE_CU_Array[i][5],
          PhotoCalc.IE_CU_Array[i][6], PhotoCalc.IE_CU_Array[i][7],
          PhotoCalc.IE_CU_Array[i][8], PhotoCalc.IE_CU_Array[i][9],
          PhotoCalc.IE_CU_Array[i][10], PhotoCalc.IE_CU_Array[i][11],
          PhotoCalc.IE_CU_Array[i][12], PhotoCalc.IE_CU_Array[i][13],
          PhotoCalc.IE_CU_Array[i][14], PhotoCalc.IE_CU_Array[i][15],
          PhotoCalc.IE_CU_Array[i][16], PhotoCalc.IE_CU_Array[i][17]);
    }

    puts("\nNotes:\n");
    puts("1.  Coefficients of Utilization calculations are based on an");
    puts("    effective floor cavity reflectance of 20 percent.");
  }
}

