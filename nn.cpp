/* 
Baylus Tunnicliff    2/13/2019

CS 475 - Machine Learning     
HW 1 - Basic Perceptron
*/

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string>
#include <cstring>

#include "mat.h"
#include "rand.h"

using namespace std;

int debug = false;
int tc = 2000; // Training count.
double spread = 1; // range of weight matrix (-spread, +spread)
double slope = 4; // slope of transfer function
// double ETA = 0.2; // for training weights
#define ETA 0.02
#define ITERATIONS 40000

#ifndef ITERATIONS
#define ITERATIONS 10
#endif

// debug/exit functions
// void e(string s);
void e(string s="Error occurred", int r=-1);

void train(Matrix input, Matrix target, Matrix &weights);
double tf( double x );
double step(double);
double eta() {
   return ETA;
   static int n = 0;
   return 0.1 / ++n;
}
// void ddebug(string s);

int main(int argc, char* argv[]) {
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
      }
   }
   int inputs = -1, hiddenNodes = -1, classes = -1;
   int target_num = 0;
   // int numscan = 0;
   // numscan = scanf("%d", &inputs);
   cin >> inputs >> hiddenNodes >> classes;
   Matrix data;
   data.read();   // Read data matrix

   target_num = data.numCols() - inputs;
   if (target_num < 1) e("target data not provided.");
   Matrix x, t;
   x = data.extract(0, 0, 0, inputs);
   x.setName("Input Matrix");

   t = data.extract(0, inputs, 0, 0 );
   t.setName("Target Matrix");

   // Create xb with extra column of -1's
   Matrix xb( x.numRows(), inputs+1, 1, "XB Matrix");
   xb.insert(x, 0, 0);
   // Matrix xb( x, "Input w/ Bias" );
   // xb.widen( xb.numCols() + 1, -1 );

   // Might have to normalize data in x, since the -1's are in xb atm, 
   //    not sure if that affects the normailization
   xb.normalizeCols();  // Normalize data before starting.
   // if (debug){
   //    // Check if data is still good?
   // }

   initRand();
   Matrix v( inputs + 1, hiddenNodes, "Hidden Weights" );
   Matrix w( hiddenNodes + 1, target_num, "Weights" );
   // v.rand(0.5 - spread, 0.5 + spread);
   // w.rand(0.5 - spread, 0.5 + spread);
   v.rand(-spread, spread);
   w.rand(-spread, spread);

   if (debug) {
      v.print("Starting V weights");
      w.print("Starting W weights");
   }

   // train( x, t, w );
   // Initialize Matrices with proper sizes
   Matrix h, y, temp;
   // Matrix y, yt, delta;
   y.setName("Output Matrix");
   // yt.setName("Yt Matrix");

   if (debug) {
      cout << "Beginning training on input data:" << endl;
      printf("inputs: %d, hiddenLayers: %d, target_num: %d\n", inputs, hiddenNodes, target_num);
      xb.print("data w/ bias looks like:");
      // data.print("Beginning training on input data:");
   }

   // Train
   for( int i=0; i < ITERATIONS; ++i ){
      /* FORWARD */
      /* h = xb . v */
      if (debug == 2) {
         printf("----------------- ITERATION %d ----------------\n", i);
         v.print("hidden layer weights");
         w.print("weights");
      }
      h = xb.dot(v);
      h.setName("Hidden weights, i think?");
      h.map(tf);

      /* hb = Map[Append[#, -1] & , h] */
      // Append -1 column to h matrix.
      Matrix hb( h, "Hidden weights w/ bias" );
      hb.widen( hb.numCols() + 1, 1 );
      // Matrix hb(h.numRows(), h.numCols() + 1, -1, "HB Matrix");
      // hb.insert(h, 0, 0);

      /* y = hb . w */
      y = hb.dot(w);
      /* y = Map[tf, y, {2}] */
      y.map(tf);
      if (debug == 2)
      {
         // If verbose debug active.
         y.print("\ny before dy assigned");
      }

      /* BACKWARD */
      /* dy = (y - t) y (1 - y); */
         /* (y - t) */
      Matrix dy( y, "Error matrix" );
      dy.sub(t);
      if (debug == 2)
      {
         // If verbose debug active.
         dy.print("\n(y - t)");
      }

      /* (1 - y) */
      // Matrix ytemp(y, "(1 - y) matrix");
      Matrix ytemp(y.numRows(), y.numCols(), 1.0, "(1 - y) matrix");
      if (debug == 2)
      {
         // If verbose debug active.
         y.print("\ny");
         ytemp.print("\nytemp");
      }
      ytemp.sub(y);
      // ytemp.scalarPreSub( 1 );
      if (debug == 2)
      {
         // If verbose debug active.
         ytemp.print("1 - y");
      }
      // These are probably dot products, not multiplications
         /* dy = (y - t) y */
      dy.mul( y );
      if (debug == 2)
      {
         // If verbose debug active.
         dy.print("\n(y - t) y");
      }
      /* dy = (y - t) y (1 - y) */
      dy.mul( ytemp );

      /* dhb = hb (1 - hb) (dy . Transpose[w]); */
      Matrix dhb( hb, "Error in hidden weights w/ bias");
      // Matrix htemp(hb, "(1 - hb) matrix");
      Matrix htemp(hb.numRows(), hb.numCols(), 1.0, "(1 - hb) matrix");
      // dhb.print("Hidden layers error");

      // htemp.scalarPreSub(1);
      htemp.sub(hb);
      // Matrix (dy, "(dy . Transpose[w])");
         /* (dy . Transpose[w]) */
      Matrix ydotw = dy.dotT(w);
         /* dhb = hb (1 - hb) */
      dhb.mul( htemp );
         /* dhb = hb (1 - hb) (dy . Transpose[w]) */
      dhb.mul( ydotw );

      if (debug == 2) {
         // If verbose debug active.
         y.print("attempted guess");
         ytemp.print();
         t.print("true targets");
         dy.print("\nSupposed error matrix");
         dhb.print("Hidden layers error");
      } 
      /// /* UPDATE */
      temp = Matrix( hb, "(Transpose[hb] . dy)" );
      temp = temp.Tdot(dy);
      /* w -= eta*(Transpose[hb] . dy) */
      // w.sub( temp.scalarMul(ETA) );
      w.sub( temp.scalarMul(eta()) );

      /*  dh = Map[Drop[#, -1] &, dhb] */
      Matrix dh(dhb, "Error in hidden weights");
      dh.narrow( dh.numCols() - 1 );

      /* v -= eta*(Transpose[xb] . dh) */
      temp = Matrix( xb, "(Transpose[xb] . dh)" );
      temp = temp.Tdot( dh );

      // v.sub(temp.scalarMul(ETA));
      v.sub(temp.scalarMul(eta()));
   }

   if (debug)
   {
      cout << "Training Finished:" << endl;
      // data.print("Beginning training on input data:");
      v.print("Ending V weights");
      w.print("Ending W weights");
   }

   ///// Training done //////
   // printf("BEGIN TESTING\n");
   // int rows=0, cols=0, readnum=-1;
   // readnum = scanf("%d", &rows);
   // while (readnum != EOF)
   // if (readnum != 1) e("Failed")
   x.read();
   // Print test case
   // data.print();
   
   // This has to be this way to trim off the targets from the input matrix
   xb = Matrix( x.numRows(), inputs + 1, 1, "Test Case");
   xb.insert(x, 0, 0);
   // xb = Matrix( x, "Final Answer" );
   // xb.widen( xb.numCols() + 1, -1 );

   /* h = xb. v; */
   h = xb.dot(v);
   h.setName("Hidden Weights");
   // Transfer function
   h.map(tf);
   // Add new column of -1's
   h.widen( h.numCols() + 1, 1 );
   y = h.dot(w);
   y.setName("Outputs");
   // Transfer function
   // y.map(tf);
   if (debug) {
      y.print("Results before the step function");
   }
   y.map(step);
   // y.print("Does this look like the output?");
   if (debug) {
      // Redirect stdout.
      FILE fp_old = *stdout;                 // preserve the original stdout
      *stdout = *fopen("tmp.txt", "w");    // redirect stdout to null
      y.print("");
      // HObject m_ObjPOS = NewLibraryObject(); // call some library which prints unwanted stdout
      *stdout = fp_old;                      // restore stdout
   }

   Matrix result(y);
   result = result.joinRight(t);
   result.printfmt("Est. and Target");
   // Matrix output(data);
   // output.insert(y, 0, inputs);
   // output.print();
   if (debug) t.print("Targets");

   return 0;
}

/////////////// Functions //////////////////////////

void train(Matrix x, Matrix t, Matrix &w)
{

}

double tf( double x )
{
   double temp = 1.0 / (1.0 + exp( - slope * x));
   return temp;
}

double step(double x)
{
   if (x > 0.5) {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}



////////////////// Private Functions //////////////////

// void e(string s) { e(s, -1); }

void e(string s, int r)
{
   cerr << s << endl;
   exit(r);
}

// void ddebug(string s)
// {
//    // Prints iff debug == 2

// }
