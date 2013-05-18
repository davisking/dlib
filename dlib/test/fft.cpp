// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/matrix.h>
#include <dlib/rand.h>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.fft");

// ----------------------------------------------------------------------------------------

    matrix<complex<double> > rand_complex(long num)
    {
        static dlib::rand rnd;
        matrix<complex<double> > m(num,1);

        for (long i = 0; i < m.size(); ++i)
        {
            m(i) = complex<double>(rnd.get_random_gaussian()*10, rnd.get_random_gaussian()*10);
        }
        return m;
    }

    const std::string get_decoded_string();
    void test_against_saved_good_ffts()
    {
        print_spinner();
        istringstream sin(get_decoded_string());
        matrix<complex<double>,0,1> m1, m2;
        matrix<complex<float>,0,1> fm1, fm2;
        while (sin.peek() != EOF)
        {
            deserialize(m1,sin);
            deserialize(m2,sin);

            fm1 = matrix_cast<complex<float> >(m1);
            fm2 = matrix_cast<complex<float> >(m2);

            DLIB_TEST(max(norm(fft(m1)-m2)) < 1e-16);
            DLIB_TEST(max(norm(m1-ifft(m2))) < 1e-16);

            DLIB_TEST(max(norm(fft(fm1)-fm2)) < 1e-7);
            DLIB_TEST(max(norm(fm1-ifft(fm2))) < 1e-7);
        }
    }

// ----------------------------------------------------------------------------------------

    void test_random_ffts()
    {
        print_spinner();
        for (int iter = 0; iter < 10; ++iter)
        {
            for (int size = 1; size <= 64; size *= 2)
            {
                const matrix<complex<double>,0,1> m1 = rand_complex(size);
                const matrix<complex<float>,0,1> fm1 = matrix_cast<complex<float> >(rand_complex(size));

                DLIB_TEST(max(norm(ifft(fft(m1))-m1)) < 1e-16);
                DLIB_TEST(max(norm(ifft(fft(fm1))-fm1)) < 1e-7);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    void test_random_real_ffts()
    {
        print_spinner();
        for (int iter = 0; iter < 10; ++iter)
        {
            for (int size = 1; size <= 64; size *= 2)
            {
                const matrix<complex<double>,0,1> m1 = complex_matrix(real(rand_complex(size)));
                const matrix<complex<float>,0,1> fm1 = matrix_cast<complex<float> >(complex_matrix(real(rand_complex(size))));

                DLIB_TEST(max(norm(ifft(fft(complex_matrix(real(m1))))-m1)) < 1e-16);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    class test_fft : public tester
    {
    public:
        test_fft (
        ) :
            tester ("test_fft",
                    "Runs tests on the fft routines.")
        {}

        void perform_test (
        )
        {
            test_against_saved_good_ffts();
            test_random_ffts();
            test_random_real_ffts();
        }
    } a;

// ----------------------------------------------------------------------------------------

    // This function returns the contents of the file 'fft_test_data.dat'
    const std::string get_decoded_string()
    {
        dlib::base64 base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;

        // The base64 encoded data from the file 'fft_test_data.dat' we want to decode and return.
        sout << "gO1l2wKz8OsyeYMPYcGx6QdBG65vnrB+omgAJ7Bnsuk9vkTw/Y9Y/UZEFXhVf6qnq92QHPLV16Fo";
        sout << "a+IUHNTjoPAfBOTyfb8QRcTj9SaWpxA65+UCJ+5L6x/TEyPKDtB23S0KRpRSdfxBSW9/rnUrkIv7";
        sout << "6i6LWcxKzdsw2WGsRCX1k3t0adQW49m/yb8LV9Loqs7/phzY7HkJ4D2PLtpc6Wyk1qG/h6KQ7nkF";
        sout << "GFkHIoh+xKXhHpqWaSofx8H8m/++H++g0VSPqfQ1ktFz+K8UtiGoyR2GqpP+br47YLXG3WqVU5Km";
        sout << "Di3+IjQoBH2m4jykD926aRvdRrgUH4gZunokl+U6shv20Zm0NL8j4A46/2f++YPGCVBNJJmcJdI7";
        sout << "9RlPL9SFbJ8rnH5bbLvZ2pKZmmbeZN78yzLUhdGwn4DGpf/Zo1fU2YPUjVKkwY6olW4w3tiBl05a";
        sout << "cS1HwBeQjnajqsXNyudbrBkM1Z9XiwM+J5iMsu5ldaJ8iLn30W2Te2RnZhJRHO8MgL7Fn1j0n0Qb";
        sout << "8dB+6aQYv0l/5LQkr5SX6YSRYX5b5rnqhi8IzJKms6dzoyBm97IGTm8pRxtLXcmsk1MvJcHF2gl2";
        sout << "CslQazsl5iIS6fMxEodmlMdwdfIpp/6MqmeIydSHwdyJJZnNPl2p5X+Il5egmwdaSoDQNphPfTaQ";
        sout << "R0Xh3xqsZKgHLKxB14Rsf/R7Eu9ZASTByX3UrEHsSzLSUo9/G+tS3n1iC30Liusksh2Wkt+/QtDy";
        sout << "A1ZX31H5OlSFwCYC/TYitwyl4U9k7WhHBDoT7MdmVTYQEK1dK48nwOhnZa9prE8n3dD40CCe25q3";
        sout << "Qo4VVYc5tBWu1TfTbshvkmHAcp3Gyw/caqq6jdq5Z2BD1b67i/bY66xhmowOFS8xeA7v6tKdkvpp";
        sout << "Rk8FegzVdB72wpw3872T4K+eplMDcCPGkwIieF5pZStWxhGsNOC0p2wvpFvTpQgfNOGUvRt69hsd";
        sout << "xaUEYlWZcY3sfsiOwPGgBUEEv6b+8W7+8Ddj8Nx4wG+bdWozphfz7THbmOeaDM63imIEHmJbZ47I";
        sout << "QgoyzFD5WoWtZ1wMEv4LL+a63B3FzBcvPvdPaa2QEmyiK9yN7GEePs2Fv2A3ymhGw5NeR1dOzAjz";
        sout << "lEQW01p8opk/dpyLO18zj8d+Hn4EnJkKD0p1u+XuLRda8AnRu/WmSOOpyG5EUrUoEyuvbECLbY9X";
        sout << "3AMgzkbxltmZlkyOOwfCGM0yumGYKdz0aGKdyid7ddLMTpQ908dCNLyRgTybdZG9137PQirgX5+O";
        sout << "08T/+L4EIyyrslOYxpUaLm2ASnSUgiivoIJvfnu8IeH2W9fPupY89ioXIYuwZU8f9FDCA9z7peQw";
        sout << "9H6l4PDdDrB7nwQhncpV9FYLkHQLbSgE1VD+eL6Y2k48pI2zUndcoHEZW72NcmK6E8fDvfgbKkYD";
        sout << "m02RiGuj4tvEEsIVuVa29Q0JGO+37n7Mlz7+RMcUMo1pLnh+jibas6R+1LCy7b4ubiKMFB1gvjut";
        sout << "gjMABy1dJxSOdb9xUa0K/Alwwu3MxdkrbTxwqkn0C2JnVV7z9S2I+PWcfZKzcpg8Itzh/ON6I/DE";
        sout << "EGK3s39XhLI2xPg3PE9R9QMaisqxb3FeP1NkBXrLQtuQfrSk+KZk6ArQWVgtem799fxgipQsa5RH";
        sout << "z2Dq9t+pJzNGUnWg5PWzaAY3lWMscn+BIRhsZfDJ3QBtS9Vmib8r2dtYwXi/Q+FhnAHFfcXbhDC3";
        sout << "GHn16aP2PY1sw8KMtfPRAcqY8Ylbr9EQXjWoIYUs0YyX2Ks8ZgibunTPFz/Wu98RVYswMtjubFaJ";
        sout << "jb0pK9S6qoe/w10CAAHqoAfca7uMOxw9trZZmjCf5vF4leH/nDgsNjesYn21rE6rLhSbg8vaZXo5";
        sout << "I/e1uhZlRz4ZNnMlZSnL70Jt0IjuR0YNphCsGZjmvvZ4ihxrcLrHvAcSTJuqW5EARtvjyQWqBKSP";
        sout << "5XhlkrI73Ejvy+Lhv6n6O7+VrfWa/tGRuvvAToS1wPOP1T2oniDXsNlD0QbMnCao+dTWgkTDiNTk";
        sout << "sFxsoN8YjwHqYUAp+hfnu1Vh2ovyemAUmo87vuG7at6f8MgFSuZffmBkGuijKNDDy7OrHoh7+5/+";
        sout << "aOkcvp0pW3ONZ4l6peRNvzaW5DEBTvcZGvRwVCHWII1eGpzeJKaHWvDfLqjaPkFrG5pR7SGCY/9L";
        sout << "73W2U0JCe2h7VjWbCM7hdvJEgYi/mEarVQpt+0P834es6Rm9rsMCbgbrWl7uv35+LVMTHU29Oxln";
        sout << "bDzBUJQs5KIA81IWR3R7D+HuJvpMkAYMF73c1owI7K74SBOsTq1ayC81aNlK7YwOOjZyBqwsQ5sy";
        sout << "zZi0k9AcKRGmTC323o7Tp/n/gkAU3NObTnqPEJitjGloXqrhPvorixBhHXSZy+wgL5R+04KiF1uU";
        sout << "LEFOzJ0zKUMstTB+fgC7D6ZnVEtUq3HEYnmaRRwEhRSgMTLXE8VvnOdo802pMVN5GMCkH299rJm5";
        sout << "Ina8mTwlC9JrNuYHot5KK/Gny4KPyUeS51cifByPwroemwBHe9EmKCkcEJPoDpG3QMaV36aopyJl";
        sout << "GwhZxaZSqbut9XSWr0IMxHUkFeslRB+n/7Vx+xWpDNjQ7JA5S/B0ZW+YBQPcjA3sRQTey25JD4Jy";
        sout << "RsULxNY5e3mjn59fI8OpBOYfNPTt2Jzppm1GDpym0LHuz7KZ6xk6QAyogk+HMjC/5RcQA7zJWDRM";
        sout << "dXC4CXUjrBxVzmm/YHXv76LrsaFdzJgn+/qzlM6IvIgicMhcJl+hA1swTkgcw6JRalJiDqnvapKP";
        sout << "V+T+/X5PSNMswgZURHQJ2l0PkMrUT909pBOC9t4GCsK8k4rYS2o0I0UYfcpm4jMRU5X34zlT8Qv+";
        sout << "GV3mA0oGq1U2dJwArlPX3gI5sZ2Jsw7Qa5edvQNG5GoRb2j2Muo4AkZXXjbx0KEa5leLIhVL4BAE";
        sout << "2GTdbL7T8hUGY3QlRQGwSVAytjUfXg4jCyn9w6ZbxUOu5MDBuCEtrhRSJNKuBLInK3Bh+fr2FshC";
        sout << "T1eDtIFE2EDEaSbLj4NCNWpTFdKMXZ9CQg2VtoVOIJfgKzqAjjcWX8kqWpMFlQgtdTfIqN7gnFit";
        sout << "do/FO0OzLghevyexHdl+Ze+MjITKOF0mTPPMkcIYcINIR1za6q3rLDZg03+GouzYhL8lwM3WAnkg";
        sout << "Qg+NM6reQATKFK3ieOxacZYnIwOR/ZMM/lO/rHY/ZbdAnJHbMBWwRtK1vDi+o+ZgS7EgsDpsmz/l";
        sout << "PguXPK0Ws51OUhIJJ5YDBv+nVPJabxOYV3dU0z49xFpxNTW9pTISo8mKZvLp2D765kExGJ9YKoAx";
        sout << "Hfi6WEg3pFS9YQLNhOZjE4bQThugIWXhi+2OPgqUIUoV5ctSnP5Lv+xhbkZfjnQQQQffrrU4peSz";
        sout << "6CuNEVLuNuG/mc3WEDZwf1HxYv3u9pr7A79QG0EROf23zPzaf5biE9e9xH+ruPApRHM58H2RpxXU";
        sout << "RlkYnfoAUqyvT3Lhhk6ngv8Axhi4otvz7sRiXQmZO7mtzWzsCTkCJoziwRKlD6P6LYnbm4fRYP1M";
        sout << "MvOuW3NhsQNrsDtgMuvqiVQpRzg157ES1i1qnTjJxTD5emK1RljuQEetbGksyetctWdWiEd8ZfSh";
        sout << "DHBJC2FLucmkMt0LHsVPnk4ni055uMRdKPRKjTE2MjpEsxR52xiWR3MtwXiEhH9fZnUl1IdBl3PG";
        sout << "TfLiZ286m4ePm6JOgNM1chtZir+q8pr4ghk/xycWvHmgkqT9dQcFP8iEtlVLCS22/2mS79cTev2r";
        sout << "yE90otp9vibcTnpORzrnLrMhhpmYRTxRjRaHGhwdJYluARJFBBVTMEenK2ubdLOJ8skZjLzPv1dt";
        sout << "9IrO1sNUwrMpEie8PG7D7DzQ7//jdlC/HUZaGKrwj5aMUULi+ZYiBLYoeL4N8ozAK1u3KtXLKlRE";
        sout << "3Akys4Py8+CmrY5qaaDOXZvwl3FF3skmGhx5KValRXrbndqr3Cks0hXglHgNonZh795galZwu0Jp";
        sout << "ww/mTQLCV0djTdEfjXBUnP49zyGXWWsEsl2jfqEAfBDcT4+mMzAUtBSwwPJYXXAJQz45R32MThNb";
        sout << "k21X+rw63QJe0zIbOJepHz3jaedMkj8GKNYBjqzibNqfYelunBUqW0bpi81HYdN5OFY/3GNKgygG";
        sout << "4R5HJaP+x9e1HxehpI/4pKFC+TAIb29uSV5GtkNTb1fYLm0kjeCZNA5GKtf42gBY52N6STl+lcI0";
        sout << "gD+jJ/ogknne3sRtEJEtCFFe1c50oikyJamQbeUO1PcDUBt8Phl1rI/p4PTP+H686usJVhSDY+b5";
        sout << "9CdS6F7XSSDiXlpFl+Esex30fRZ8zAQsTo9oN0sSZUUJKcyVk0dCqw2mHWPpyM6hYKQ3ij1nYjYl";
        sout << "3PzRfFMlu+dgStcBn70jvlEv5pOWXb2OqrN9nJtb29n8jrB2K2nlbcYoPPiQ3yXk+Wpom82LoT5W";
        sout << "F9NeNwwAB4EDWtB96OU6noW8NHJj7NiADQJGvQpk/3JzIzeBJQCxULYJMRJdBKf61+24F791reHa";
        sout << "qrH+rLUrrv05dIPDTUvGW5LQLTTFFa59OmMIu7WJE7Ln6gMIwDw3FXnGFzaWnHlHL/9jJ0zM1FQL";
        sout << "kfK4wTd++GbeI0gsnXWFK0N0kV/FiHm++J4udWwIXZxH7qZCHtwlT/5oGDVujtAtOPag+txUrjVc";
        sout << "G4iLeiPbV/2Vfc2D1oV5/yyXDDii9qrLH6SOOfgvdiJZr7X3uMUIDGO75x5wBDSxr9t3I2CrX2dM";
        sout << "M6kD7U1+bf5QVRbkh3Us4NAhFVnLNEcrm0x9Yx0wRmxPKgJeGGbWi7/BHi8ShIFllizuxuMyfypC";
        sout << "hhzSlxxbYAQwtcC3cHEnyYZAO7HC6hyke+HQJfxAmKyfguGtzEzsiG18XJVruwz7IoOpZS/O71zy";
        sout << "Nv+T8trOhy59ZUAgIsCAAQJYEBWl/T/qFtkE+tITbVRKtHjbxHSeN12OnHFRoKguJYaakTo4qLs0";
        sout << "fr4E4nZUMfjdF7oI7YutegY9TkiJ9ujLJw4pfY1XRtPrRukEl8orypWXq0gErnYO/RVtK3XImrDp";
        sout << "LY5sXH5pNzkqVH9VCl6lh9sg2HWjNwv9bDcDlIhvTL19Mx9yUtx/iQtG/OKy22tW6ByahPNnMNtA";
        sout << "tBVB38RLf6eJr68mhn10Qg68cXxVL7/zEIZd9rUaCo8xCzeFblDNErKfG02JJ6fbQ6M6ZdNez7Q0";
        sout << "x2IYbz2DEk0wHmR7OtA/oTFMvJlyMt+dDWTEpHnvqkbe+veENpxn2WWy3UsumkvhhtzzmLxyD6Sh";
        sout << "mMbMPwgUjvMG51JfRrgMfJzT49z0sebSfzvid/9QV4lNkR7s9nfUJEwAued4S4klRy3LiFdQhjQR";
        sout << "FOZZNqUge8vxVOzVCfS+xsjvnGrd7azt7LJg6wPXFgPfeE2bRlx+8AoRFG7SUpudmm/bkNw+uNgS";
        sout << "YRdaH8p16RyNoMlSfi/7BNDhtKwrl202pVuCqhFey0mPYehYee2HhLZs6ph+HKMYy8lZ/ac1Q17d";
        sout << "1tcI4WH0Hz0B/3GWl8xWfoq2OO40EIjuCPNhk70MpiytWXggJrKoKPu52GOqTU8+jZ6F+u6U2muZ";
        sout << "6QZLYXDwPaNz/lq5U4ACw767DkhUHd1/h0g6r/RwtLKxdrzYldQto99TAMmHc+z9aIciTv7kl/Gs";
        sout << "WA58nI8aODhwjIkOGaExdlR1k/3JR2tAAj5vRzYlJeakhAA82pA+8xMPZr3HRlQx4DlEjH1spAA=";

        // Put the data into the istream sin
        sin.str(sout.str());
        sout.str("");

        // Decode the base64 text into its compressed binary form
        base64_coder.decode(sin,sout);
        sin.clear();
        sin.str(sout.str());
        sout.str("");

        // Decompress the data into its original form
        compressor.decompress(sin,sout);

        // Return the decoded and decompressed data
        return sout.str();
    }

}



