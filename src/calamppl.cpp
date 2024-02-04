// *********************************************************************
// Calculate amplitude calibration constant for SLC files
// ---------------------------------------------------------------------
// AUTHOR    : Andy Hooper
// UPDATE    : Rahul V Sharan
// ---------------------------------------------------------------------
// WRITTEN   : 04.08.2003
//
// Change History
// ==============================================
// 01/2009 MA Deprication Fix
// 03/2009 MA Fix for gcc 4.3.x
// 01/2011 MCC neglecting pixel with  zero amplitude
// 12/2012 AH Add byteswap option
// 02/2024 Parallelize slc amp calculation
// ==============================================

#include <iostream>
#include <algorithm> 
#include <fstream>
#include <complex>
#include <vector>
#include <string>
#include <omp.h>
using namespace std;

bool invalid_argc(int argc)
{
    if (argc < 3)
    {
        cout << "Usage: calamp parmfile.in width parmfile.out precision byteswap maskfile\n";
        cout << "  parmfile.in(input) SLC file names (complex float)" << endl;
        cout << "  width  width of SLCs" << endl;
        cout << "  parmfile.out(output) SLC file names and calibration constants" << endl;
        cout << "  precision(input) s or f (default)" << endl;
        cout << "  byteswap(input) 1 for to swap bytes, 0 otherwise (default)" << endl;
        cout << "  maskfile   (input)  mask rows and columns (optional)" << endl;
        return true;
    }
    return false;
}

vector<string> load_paths(string path)
{
    vector<string> paths;
    std::ifstream file(path);
    if (file.is_open()){
        std::string line;
        while (std::getline(file, line)){
            paths.push_back(line);
        }
        file.close();
    }
    return paths;
}

void cshortswap(complex<short> *f)
{
   char *b = reinterpret_cast<char *>(f);
   complex<short> f2;
   char *b2 = reinterpret_cast<char *>(&f2);
   b2[0] = b[1];
   b2[1] = b[0];
   b2[2] = b[3];
   b2[3] = b[2];
   f[0] = f2;
}

void cfloatswap(complex<float> *f)
{
   char *b = reinterpret_cast<char *>(f);
   complex<float> f2;
   char *b2 = reinterpret_cast<char *>(&f2);
   b2[0] = b[3];
   b2[1] = b[2];
   b2[2] = b[1];
   b2[3] = b[0];
   b2[4] = b[7];
   b2[5] = b[6];
   b2[6] = b[5];
   b2[7] = b[4];
   f[0] = f2;
}

string run_calamp(string path, int width, string prec,int byteswap)
{   
    complex<float> *buffer = new complex<float>[width];
    complex<short> *buffers = reinterpret_cast<complex<short> *>(buffer);
    int linebytes;
    if (prec[0] == 's'){
        linebytes = sizeof(complex<short>) * width;
    }
    else{
        linebytes = sizeof(complex<float>) * width;
    }

    float calib_factor = 0;
    double sumamp = 0;
    double amp_pixel = 0;
    long unsigned int nof_pixels = 0;
    long unsigned int nof_zero_pixels = 0;

    ifstream rslcfile(path, ios::in | ios::binary);
    rslcfile.read(reinterpret_cast<char *>(buffer), linebytes);
    ostringstream stream;

    if (!rslcfile.is_open()){
        cout << "Error opening file " << path << "\n";
        stream << path << " ";
        return  stream.str();
    }
    while (!rslcfile.eof()){
        for (int j = 0; j < width; j++) // loop over each read pixel pf the buffer
        {
            complex<float> camp;
            if (prec[0] == 's')
            {
                if (byteswap == 1)
                {
                    cshortswap(&buffers[j]);
                }
                camp = buffers[j];
            }
            else
            {
                camp = buffer[j];
                if (byteswap == 1)
                {
                    cfloatswap(&camp);
                }
            }
            amp_pixel = abs(camp);
            if (amp_pixel > 0.001) // rejects pixels with low amplitude ~0
            {
                sumamp += abs(camp);
                nof_pixels++;
            }
            else
                nof_zero_pixels++;
        }
        rslcfile.read(reinterpret_cast<char *>(buffer), linebytes);
    }
    if (nof_pixels != 0)
    {
        calib_factor = sumamp / nof_pixels;
    }
    else
    {
        cout << "WARNING : SLC " << path << "has ZERO mean amplitude \n";
        calib_factor = 0;
    }

    rslcfile.close();
    stream << path << " " <<calib_factor;
    return  stream.str();
}
bool compareFunction (std::string a, std::string b) {return a<b;} 

int main(int argc, char *argv[])
{
    if (invalid_argc(argc))
    {
        return 0;
    }

    int width = atoi(argv[2]);
    string outfile = (argc < 4) ? "parmfile.out" : argv[3];
    string prec = (argc < 5) ? "f" : argv[4];
    int byteswap = (argc < 6) ? 0 : atoi(argv[5]);

    vector<string> paths = load_paths(argv[1]);
    vector<string>::iterator ptr;
    vector<string> calamp_data;

    #pragma omp parallel for
    for (ptr = paths.begin(); ptr < paths.end(); ptr++){
        calamp_data.push_back(run_calamp(*ptr, width, prec,byteswap)) ;
    }
    sort(calamp_data.begin(),calamp_data.end(),compareFunction);

    ofstream outsteam(outfile);
    for (ptr = calamp_data.begin(); ptr < calamp_data.end(); ptr++){
        outsteam << *ptr << endl ;
    }
    outsteam.close();
    return 0;
}