/* 
Baylus Tunnicliff    3/26/2019

CS 475 - Machine Learning     
HW 4 - Principal Component Analysis
*/

#include <iostream>
#include <stdio.h>   // rewind
#include <cstdlib>   // atoi, strtol
#include <cmath>
#include <string>
#include <cstring>

#include "mat.h"
#include "rand.h"

using namespace std;

int debug = false;

void e(string s="Error occurred", int r=-1);
void ddebug(string s) {
   if (debug){
      cout << s << endl;
   }
}

int main(int argc, char* argv[]) {
   long k = 0;
   bool isTranspose = false;
   string extType = "";
   bool iscolor = false;

   if (argc > 1) {
      for ( int i=0; i < argc; ++i ) {
         if ( strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0 ) {
            if (debug != false) {
               // Debug already set
               debug = 2;
            }
            else {
               debug = true;
            }
         } 
         else if ( (i > 0) && ( argv[i][0] != '-' ) && (k == 0) ) {
            // if argument isnt the program name, 
            //   and it isnt a flag, and we haven't found k yet
            k = strtol( argv[i], NULL, 10 );
         }
         else if ( argv[i][0] == '<' ) {
            // Input is about to be redirected from file
            if ((i + 1) < argc) {
               // We haven't run off the edge
               string filename = argv[i + 1];
               int l = filename.length();
               if (l == 0) e("Couldn't get length of filename");
               // npos means it will get all remaining characters in string.
               extType = filename.substr( l - 4, string::npos );
               if (extType == ".pgm")
                  iscolor = false;
               else if (extType == ".ppm")
                  iscolor = true;
            } else {
               e("Not sure why but we ran out of arguments looking for input file after input redirection...");
            }
         }
      }
   }

   Matrix picOrig("Original Picture");

   if (extType == "") {
      // We need to look at the first couple things before we let the matrix read in the data.
      // This is so we can check what file extension we need to use.
      char magic[3];
      // get magic number of file
      if (fscanf(stdin, "%2s", magic) != 1)
      {
         printf("ERROR(): unable to read file magic number for file named.\n");
         exit(1);
      }
      if (magic[0] != 'P') e("I am not sure what i am reading, but it isnt a supported file type");
      switch( magic[1] ) {
         case '2':
         case '5':
            // .pgm
            extType = ".pgm";
            iscolor = false;
            break;
         case '3':
         case '6':
            // .ppm
            extType = ".ppm";
            iscolor = true;
            break;
      }

      // Now that we are done with the input stream, 
      //    rewind it to prepare it for the matrix read function
      rewind(stdin);
   }

   if (iscolor) {
      picOrig.readImagePpm("", "original picture");
   }
   else {
      picOrig.readImagePgm("", "original picture");
   }

   if (k == 0) {
      // Give k a default value
      k = 256;  
   }

   // Check if we need to transpose
   if (k < 0) {
      k = -k;
      isTranspose = true;
   }
   
   Matrix data;
   // Matrix data( picOrig );
   if (isTranspose) {
      data = Matrix(picOrig.transpose());
      // data = picOrig.transpose();
   }
   else {
      data = Matrix( picOrig );
   }
   data.setName("Pic");

   if (k > data.numRows()) {
      ddebug("K is larger than R, setting K to be 0.75 * R");
      k = (0.75 * data.numRows());
   }
   ddebug("centering data");

   // Center data
   Matrix centeredData(data, "centered data (X' in reference)");
   // We dont actually need to do this because it is done for us
   //       When we use Matrix::cov() in the next step.
   Matrix colMeans = centeredData.meanVec();
   colMeans.setName("Mean");
   centeredData.subRowVector( colMeans );

   ddebug("Calculating covariance matrix");
   // Compute covariance matrix
   //    We use data here and not centeredData, 
   //    because Matrix::cov() centers the data before it does its calculations
   Matrix covariance = data.cov();

   ddebug("Getting eigen vectors/values");
   // Compute eigen values/vectors
   Matrix eVectors( covariance, "EigenVectors" );
   Matrix eValues = eVectors.eigenSystem();
   eValues.setName("EigenValues");
   if (debug) {
      eVectors.printSize();
      eValues.printSize();
   }


   if (debug == 2) {
      // Test if the eigenvectors need to be normalized.
      //    each vector should sum to 1.
      eVectors.print("Printing eigenVectors, these might need to be normalized");
      eValues.print("printing eigenValues");
   }

   ddebug("shortening eigen vectors");
   // EigenVectors should be normalized by now.
   // Now reduce eigen vectors to K largest eigenvalues.
   //   since vectors are already sorted Max-Min eigen values, 
   //   we can just shrink vector list
   Matrix shortEigVec = eVectors.extract(0, 0, k, 0);
   // eVectors.shorten( k );

   ddebug("Encoding data");
   // Now eVectors should be k x C in size.
   // X'' = X' . T(V^)
   Matrix compressedData = centeredData.dotT( shortEigVec );
   compressedData.setName("Encoded");

   ddebug("Decoding data");
   // Now we have our compressed data, time to uncompress it.
   Matrix recoveredPic = compressedData.dot( shortEigVec );
   recoveredPic.setName("Decoded");
   // Uncenter data
   recoveredPic.addRowVector( colMeans );

   ddebug("Calculating distance per pixel.");
   double dist = recoveredPic.dist2( data );
   double distPerPixel = dist / ( recoveredPic.numCols() * recoveredPic.numRows() );

   Matrix outputPic;
   if (isTranspose)
   {
      outputPic = Matrix(recoveredPic.transpose());
      // data = picOrig.transpose();
   }
   else
   {
      outputPic = Matrix(recoveredPic);
   }

   string outputName = "z";
   if (extType == "") extType = ".pgm";   // If we never found the extension type, set default.
   
   outputName += extType;

   if (iscolor){
      outputPic.writeImagePpm( outputName, "This file is in 8 bit color" );
   }
   else {
      outputPic.writeImagePgm( outputName, "This file is in gray-scale" );
   }

   ddebug("Printing results");
   // Output results.
   data.printSize();
   colMeans.printSize();
   eVectors.printSize();
   eValues.printSize();
   compressedData.printSize();
   recoveredPic.printSize();
   printf("Per Pixel Dist^2: %.5f\n", distPerPixel);

   return 0;
}

/////////////// Functions //////////////////////////


////////////////// Private Functions //////////////////

// void e(string s) { e(s, -1); }

void e(string s, int r)
{
   cerr << s << endl;
   exit(r);
}
