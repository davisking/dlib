// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include <dlib/random_forest.h>
#include <dlib/svm.h>
#include <dlib/statistics.h>

#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;


    logger dlog("test.random_forest");

    const std::string get_decoded_string();

// ----------------------------------------------------------------------------------------



    class test_random_forest : public tester
    {
    public:
        test_random_forest (
        ) :
            tester ("test_random_forest",
                    "Runs tests on the random forest tools.")
        {}


        void perform_test (
        )
        {
            istringstream sin(get_decoded_string());

            print_spinner();

            typedef matrix<double,0,1> sample_type;
            std::vector<double> labels;
            std::vector<sample_type> samples;

            deserialize(samples, sin);
            deserialize(labels, sin);

            DLIB_TEST(samples.size() == 506);

            random_forest_regression_trainer<dense_feature_extractor> trainer;
            trainer.set_num_trees(1000);
            trainer.set_seed("random forest");

            std::vector<double> oobs;
            auto df = trainer.train(samples, labels, oobs);

            DLIB_TEST(df.get_num_trees() == 1000);

            auto result = test_regression_function(df, samples, labels);
            // train:    1.95064 0.990374  0.92738  1.04536
            dlog << LINFO << "train: " << result;
            DLIB_TEST_MSG(result(0) < 2.0, result(0));

            // By construction, output values should be in the span of the training labels.
            const double min_label = min(mat(labels));
            const double max_label = max(mat(labels));
            for (auto&& x : samples) {
                double y = df(x);
                DLIB_TEST(min_label <= y && y <= max_label);
            }

            running_stats<double> rs;
            for (size_t i = 0; i < oobs.size(); ++i)
                rs.add(std::pow(oobs[i]-labels[i],2.0));
            dlog << LINFO << "OOB MSE: "<< rs.mean();
            DLIB_TEST_MSG(rs.mean() < 10.0, rs.mean());

            print_spinner();

            stringstream ss;
            serialize(df, ss);
            decltype(df) df2;
            deserialize(df2, ss);
            DLIB_TEST(df2.get_num_trees() == 1000);
            result = test_regression_function(df2, samples, labels);
            // train:    1.95064 0.990374  0.92738  1.04536
            dlog << LINFO << "serialized train results: " << result;
            DLIB_TEST_MSG(result(0) < 2.0, result(0));
        }
    } a;


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------


    // This function returns the contents of the file './housing_data.dat'
    const std::string get_decoded_string()
    {
        dlib::base64 base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;

        // The base64 encoded data from the file './housing_data.dat' we want to decode and return.
        sout << "AvlDWRK3FGmPtCL8V/RoXQLzsOKukA0zQosjXGZPM6yNf1U5LjdhRVloILZ5baZq5Tj0hLQGf1JY";
        sout << "ggTd0DEbBez7lzvZ6hOBABkJ6U0aZAEeIMUL/K19h5gHhpwdWvcolJA+VbQVD4acLQFxcIgsgN8N";
        sout << "2zjQsQCUkHGSGyop7/xrVZAJS4Nwy8qMWJyjgc/9mZXdDsaPWtSVTeoxpb3d94bDYdrEQ8T0Z6N1";
        sout << "fCXCQo/3bus7NA+FJoOtw23DHMsYnsv+tfvaNzzzX7lc0cRPe8Pi5q9JDBs4Gc5bPh3Hw8QJz3n5";
        sout << "beGJpU1KRHu9vq1zgFuavXvyZ1HQVHLj/yJRm4lL5eUv7CQeULGP9UmIkvVCJTZ4uw6mIFJyPYjN";
        sout << "jjygWMfJ1dnlGpI/ZlaVNJaTEB28tNymV0UKeKb/Sg42s9+wNNQzWMBm5TRvqmplf/0Gx+4Tcmcd";
        sout << "FtUb1pgkz6OB59Ko4L2hsxVQYYZHGpQt6QBMiubW7GVe2g4yWjRhqlQTh2sjceGRi26SFOrD+gnH";
        sout << "9xZlbyKdlKlcT3nVfcKYziLKAjbmr5QIu+W6WR7M+p90CHkDrjkVK0WTSc23kOhuua5feG54aoht";
        sout << "hcViWRpASVPXsKKJcl2yTlZ02uFQak5Lid/znCDmWQk0fjfzEZZNzgoCNVi8xCx68lH4Mjm3MdF4";
        sout << "ExZqX0jsNlxDOwKp7TYIbhfY4/XdzFSi20CtxXbB3knkuggb+ru/u/6lh8rmqwLjsANqb2CWG0EH";
        sout << "32i1/gmtlY54kAYG58GWB89klTnUwRImQv/QjJBrf8TwK0jUOjkrnOWgKWwNsK0jba54750QPal4";
        sout << "SJFvmeBIR52/T2ZsC2iAQGQuod1IZtIfI2pn0dpkdWm/Y3JdelJ/EADtNjJch6maVoamxKmoWlVw";
        sout << "TRoHTIzdsJKtQiewn21H8bFX7HzZ1yE3/iuSEeLPB7T7gnfrtCEEzqAWiUb/t7mXtqBqt6Kdk6By";
        sout << "JZqE0KdtJ7MJ/yV1MHQ94ExDhYI0qaItZVGwyH8ETCzr6xqmue9hPD6SRq3oy+aQKvJxOqFMcqsj";
        sout << "cL+/2S2F1frRgZDGCb6tB3TdMCJDhZoChQNmJ3hyoAdWXrPEysL0TW8LFSIItAylIRljMsXgnMRE";
        sout << "RCkfYPdbweT71k6l7FiaAsqHeKh4w8CxKJkzhJeALEPLz4QvqFt6/DoFmhKTrX4xUk3M/Y+dU/LY";
        sout << "3B/S+e6v9cWuluEGXDzgj/LKBeruPk7hcnhIilMmd3D8sew3tvOdIowxmM67lqW6fwExcK6oKPlT";
        sout << "aDZcttWKTndVEKsUvnZr/PQ8sta49+GGSfkXw/MS0TAjTQ0Wck8wSJ2CMoUGBmVSKLSwEWvdAqdo";
        sout << "lQLDxAVayR3GeKasJshQw69o/3d4JnUOBcU5ZJM0z2D51EDQM3lvmnB9dtiJ2rcypWG53ETvQqYc";
        sout << "S3suPgaDqKmxZbRNvnfuYbG6+qPeHDN6WmdAt9Iw5XWdjlG6u8BGI6+vqY1C8J6pJ2p7ITUVBVCU";
        sout << "NRYyXVhz0PQrzy5jFwbVeRZo3IV/bkPlHV4UujcSEOi67fqk4ixOzA2JsxYzei+Rsp1ahpK3Wmuk";
        sout << "9ZEeqD1/xqqeosV5pvwcXhjjp4UJ0bGY0pEf7w9uDW0aZT+D8prSmTXGjFAQGiSBmfLFw1Yk2CyG";
        sout << "V8RG7/7uxM6qyj9LYsNTGvdyD8DvQNEJ0v7J9IwCihdJAFhuKgqxlmkJx3mz6MiiIC19CuQKR0NC";
        sout << "1/PG2wh7zEhnDwfINSR2b41ZDcm8/ky/k+xhJ6fi3ZqpSlkqyPRA/YAID9Dl7Ngn9/xeWQNXzEHD";
        sout << "pn6aPHclBaDm5keUHlcN9d+vrxrRjV/GdRhs5M2eVRl47djEiYTaQ3Ua9Lg0oWBXYFkC1ncNhsIT";
        sout << "tFLwx1Fc89lNwN3kHW91X3g1MMDFZoyzwBan2v/3DOZpPH4U1cPr1SbAPu0HITHK6pCUtMRY2/DZ";
        sout << "9MTmm6FLeJhjiVma1+ALD4wYTNhebWkmX12jeDPT5gMyDhq3dYQKIvq83aVY1MQ2UroMDf9NVEdh";
        sout << "V94rpEjw0ewZeDKHwPazkOU4q5m69VxwkKSbNzKZ1oe0P5s7j4Z44NYP3o7Qq8MPLi9l7QVKqge8";
        sout << "6PqEdYoxGr9a/QDB7wcSdpcZTku0MippOZm1uG0zA2K6WlTmC/3lCm4m8EZBZXq6YkTjUpPreeio";
        sout << "umUsmp+XbtJ1LWK3V5hMl0R67Fa6tBd6x9bP6/FSwjeYPHj8dz3nK0VLX+NC7+NjIPBkKgAcqs+1";
        sout << "m9u/BA72sNE/cn7NgfgXPlHsho3reV8Iujq+MTN5iayeTH2fyG7XmV0RkpOY770bJEdugB4QlWWL";
        sout << "nZgYd2yyNFlBcvXRKZpoG9cTWuzdxSbJnYdgKGLuPQ0B0BYwrSLnHEM8pboXCN4uIrALW6ipmKeO";
        sout << "/S8bW4u73Bqgamnh7/pdLAoWo6EiD5C3uNrW0OYdnnSjhLHkV+krDhPQr6nAfjH/0vo+CXPDbMyW";
        sout << "DmkJVgJ/cBt+EWNyIOeBLliqbe9zY6hqzGRt4b6cp1fDH/tbYMCsxhp0LIPnDNDouUQK3j+VBB3X";
        sout << "E8NnCSzBa4SdhMNww7jbJea+RXqe+g1clXqkBf/ZitToTPmzHhYcPRAcvhLcRE0n/uWj7jbv5vOD";
        sout << "swyUpiJaXQgRG77rh89xNVqz9qio6uP2xGdlyW0IENbxMYsGq+XKxNMAMHvfRh8JwakGw7ZI/Wn7";
        sout << "8uWdjM2lNmenBCYSS9qe+2DKnqTxOSnm5ugsYr6IXftmlzev0ke2rRBfvllAv8GSY8GTJul+gbZV";
        sout << "+3Wu8xZawgLFjngRph0aq4PviIwrMS1PhE5M7pC65E40uaY+xJv4rQGNCLF3/+SLvnLfTRdH0QZU";
        sout << "r/hXG0BCcaWE4hb7HIzon9mNIZf2Eb+IWxAhUQ2/Nhe/hNTRx+DpB/8H2DurZPFK4nrOPvxmmzgA";
        sout << "3VFL0kJjNfeXGlo2sSQEM8sDecXQkl47KGWROHIaJyRZAoBOBpMHxTXs//3aqWhlOZ88pftZEmSL";
        sout << "K0sXXxS3BAKB8SLzu4VNPNdvmtT7Z4sHmfYj5NXayXS3V2d2646L2QW4jzzwHjpJ6p2/4mjKniFO";
        sout << "TSQu1wKSv16L/QVbiwvg6wYgIL8cct+XSRSUGfvVgo9lAt+OHTIL7s/9A66jqdDmK3UnHcV8XOVN";
        sout << "Wd+DnXOG7nI5shOiGHAqAAtuQZK9sAZZtyLmg68UN02ZroY0pUePwuGaBribKuKLDtcfMcDHwv0x";
        sout << "lSlCkHu9SjsC1Qswk90Yx8YddYY1ePYaoez6xUAraj+zOLNuFCZtm6hGTQq+pPZ5xn/K6zzOvaos";
        sout << "sxDaWBSNDEuOfAK2dHgctL6XKH7/kHAZxBa7CbTSe0zFuO1WbicRrqO1NpUyO9P1L82dv1VCSuyi";
        sout << "Mtj7UNnywrmZmMBf5x5yYBdVwBUKIBy/sNJdrpDHbAi6MJoYzCai8TqTulNL8jGAzkjpXYRqPf99";
        sout << "fXWRTzN1PMHjvEbNOBIX4OorGR4lj4E7i+d1DKaJjXCDgvJQTUGvRWdu7MOOkXPCmIlxFL9Wv2CB";
        sout << "LpzzNp+I3NuLg2F30ratBBLoqIrnBBXb390pABYah8bRnARCUJLjFXugVqTWoMwAsrbS6sfdFRf8";
        sout << "fKt/+Nx2vX8tRJBFFgBEbS2le05ekg7HC6egGCLImh8j8sf4gs+2xdGKXh9mnW8BrqZJvQPkeR4D";
        sout << "Fro5V/EFe7EAIXpQfMRoNpHUSyn5oPJDFYMjjc1EEO4C6qqJ29nV149m60BjWDuVK1y+mdkCvaDo";
        sout << "iZfAKI4TiDExfMpdAJM5Ti6G7pauQnW/lxGNX44JOR82strVKwsQxSecUc+uT+kYjcdV9Mx29qJC";
        sout << "qXJgi2cUhrNsVH0LIe45X3pSzfFnMQ2o+fGAURgQCSW6wToqmtHBsCorr0v32ew524X6tZz11HMC";
        sout << "7DKppzWxGTBOCBwOPrAjwlUb8zaRpj3bQGXWkvDk7E7ZUMQ6TJu0wgkuBNIPeIAh2tLfGqrqZqxp";
        sout << "Y2hM/G/qQG+Bukj8yyxitXqIwceSp3v2BnLnL/WriBpSVEm9LnjiPou/BL98WMhp13UKWe3L3XkC";
        sout << "izri1YMoCyxaQFX6RODu4We1vi/9gjamFSCAc5Tj+CmUasCJpbmkjt7Fp+v3PhXa4qpt32ZR0yuF";
        sout << "G0dowpb8WYvT3U7nWAOFKBRgj9Ri6QPlCVaUa/LnkjZ+fNzytlkzQ9TTsPpOkEeJo+nCF3cCUYBH";
        sout << "Y6lIyjcQRk9guLIgk955mBLpyjni8ZFOOsjTsW+LoOvAiZhVTGwA75/g5z6IcYLlcb0nwpZ/O2fS";
        sout << "QPFcb5V6uhi5TnQHDQGHihSU4MBo5BQfNd+VuSxliK/TVvFU0yYjqPzKxCxgBpDO8qKsPMbc2YKL";
        sout << "SFY2ygJ7PwksSIEQUum0MSEFf1ZJ3WNTajxSvFLToOkAtLpnvZlWymYkI72/Dgi7jBpfhIw1U1Td";
        sout << "tuTLc0L6IfALX2I2VL2tOhBcisUL8IRhDipxhBTRBaJYLG2RB6ICKBuAQaXf4ODAPKbLhzfRSss+";
        sout << "2VTojSwerCyQkKyoUZLR67G2ysWWLERwD1btSNH4IjaPYVEmaWk4I4F1YZhrmN3q5du7t7g3E4C4";
        sout << "/UVLrCVTQD0CnBVBB5hzMEByG/4ZhIu+JWx+jRx1288XA1k84c28NLfMnqDsLHGtVxOLFDBgwFxs";
        sout << "vD8S2E1+G4La3DQWc/X6jfkC+dtp0ihh5qQxGaGCKh0mcd3BHnNJYUSqSRQLRhOjiZBxmujGrhJG";
        sout << "oHPaUCxfgY3vl9y6KAVlcLcKTYZpmukkjxCEVOClOy2pHYivhgkO2HR7okgNGpj8qN9EcVTWPq8u";
        sout << "dbBjHLQ2GbqHamyaDJFUhJfsibyXR5ZtG2WAZDH3uXlL8AGNriBhwgGVcaRGH4sO/NmWWdM/gnap";
        sout << "6geVpginIZALN+egxDTQtxTH5qTPfkMg9tdjlX/zB7e1LbVR40waeP5PtanIvb7VU/GbVbQMEDKc";
        sout << "Lqfj3v6RyK6wX7mDvF7HWFtav8R/j9wlgf75kOiXz+2eN7GeXEmF68LqH6g4n7Ulyhq3uqszT4Jg";
        sout << "hk7ynKJoLURg5KyJPTUCvadDaiqaLH4hF2bErrQzIIbbKDCq5Cb7n0EhCZ03MjLFs9+HSa3yfaGN";
        sout << "NUS3wdGiM9x6rNaKDP6/vySXZzBvgtinskFBvb7UqCwmQ2MF1lwr0+nTNfH7R9fw3fi+tHXB6kyh";
        sout << "PovdaPH+3dfnsbqVSoJLj2OvjsFfTFQXn35xd/IW3UEdBYVSDZP8VGRnXQUSS4BbJ79VUMNOTmwz";
        sout << "CsoiZzIZNgHShekR2XKv1oXM+BheSAxK+r/d+VdPgjlkByfCwuw8iP/odUoaXzk6iTh8h1pGyESL";
        sout << "QY8mNIzzPsU39opNlK7JmOzlYG2wtCS+DcG+bw4HLJP3Or9mChHpN+V3xzL5Tsb/5fGeqcQ0hvsA";
        sout << "aXMhlsnRtxRSDkfE0s1HW0r/O63X5Hm1Yw/vJw8BzEtNYg8h3x7xvECS4vAwwuKLS30rjlVqjPqI";
        sout << "TNchzWOA98U8AoC3t0asTAaEXce8tPtLqXD0EyycoU3slyfpErU4vySzpCXtkv3BShevfZy9yhX0";
        sout << "2HG5zTc+l8GdXayf6mVSXaQ2N2OV6gCwd+hwqHjqvYSg4a0Ug+/cEw3zVi5AGiLIzTGDGsfJHE86";
        sout << "9ohKS4z7yI+doqegx0f7N95Njw315nKSZnSSf6Pa/I20SrcQabMC36H2vdv9gkOlsYlZLyCOL54P";
        sout << "ZOXlim7GgCt8LPEO6maHmQn0f5mtJAYIxMrJKoMasXvc2ZI2tktbh6bAJNfpSL0KbTfeQtaFJnVX";
        sout << "C2f8RXf61VY9rNVd+qtpNuiavf8ZuaVbSLsLzF/beAFpS4djI0Nn78CJBhZAnzhPD76byg/vXG42";
        sout << "nyD/u/FLJ3eccPnvs5umbz/gPiFk6gW+HnTXaYEdwaGWDdlr4QxvDki8Wsr0AWPlzA8D0nmkPCZW";
        sout << "EZLBUIUjnBN2K86dyqEDW8+C42vuwXfa31wkOX0/8S7FSuT1BET8HdK8fykJ8NxdKlUsIFNr9SPz";
        sout << "maMVyvkPp6IQ+DG9PBOFIaFy+zHbPzCRNNd3LTBhkQ0K6bP7u4tG6b4fdmmQzSSGsfXqEUiXkjGX";
        sout << "ge14Qh+f/2KA4TjBZDQWF5NKR6/x4lsHfj8dZDg3+fEwY2fqezjD/jptis7N/VfeSIM/3xD+gF3w";
        sout << "BqJn4Wz+ohlWucLfS0JRREnPAWfje7RQYatBkLok2Uy2hO7lgfw6ipUHNVPUw6XmK3VW+McnK0Ur";
        sout << "L4CI9LAFF+kDdBfTs8hnhmLtk6h2Sucjo1ahEBxAyUuRgqMko5Sy8Lr9Eo49KiKO+V8LpA5ZDMq8";
        sout << "iabdyb1WLFnyvE01K4uKqGHLoeh7heD7/0sAbIVySks0mv5cH46AT288mIVcHrSUurhtxawYZY4P";
        sout << "/DqW0jHbqIkZWJiOquIfeTbtgRax76gX01JSeEdL0UyPHTmoqkvMdQVwjYcIBdLrCPPWQWNjkmWa";
        sout << "XBedBCzwmp6fZX8ew5AABNCmlBBlhqNFZQ4yG91NXuiDoS25xckthn+6l7Mn0FHs9418wKqa+3eB";
        sout << "uGqiAJVwpgWx7CWhWi9MonFdA1nni9AyjubzEaUSbzjL4ghneOGC38FyEQcuKIxrqgY+ManAAdlc";
        sout << "hVaHl9Rx4r16AITagHNPLwCbbeJ7nbM+arjvU6qmK4Bg4E96IDjrrp2EQJMYZrs5+oRbpdtomWTx";
        sout << "k3hcUFCMUiulWQs/pc5bXm+Xvx0mNnpu9A4GtFMzzpKO4M9Q/mtKch9H547N8hjV4jrWsJCqplho";
        sout << "XIp9yBwPKVaEwWYpFcmpfQIJW+I56ewA0xthRBdqqqQSoS3zMJwdQEUm+XibYA9XALC2dkbTH2fo";
        sout << "H9a4ImxxquSZZpoqUuRsbsgejD2v0ynbipTQ/lNVwswk18Wma+Whg4sOdCSQIMns2QW/3quqHoqb";
        sout << "DC06jJZuQJnLNly7x48Pcus1gsWc6aVhbpl7cCLq/YUY0ACMS5toED6q+5mqakCg69pK4dm9WHIf";
        sout << "D0hRK0v/05LheyllMYmQhO98Z0Jz8IJQQsl2sZahUr4Q8oTTFt9rLRKd+onL5HwdJQDAiYB/fex0";
        sout << "3MHYPjyzK2aH6mN7qOG2VquMGb6LmOPszLSggrzFsVzPqYrzqH/6sGZPV1wMduWw/qYadGPlpzu/";
        sout << "XIgnQ2Qzb21xqwTnivx30xgWDf8kuQWDdECGT+/GPAqNx1P9RGLru+tSPe06oZ8RuoSs64zDqJ3E";
        sout << "KMmOeFAt+xQFBg6dEBs1pgfO78AfZDAbadYzvPCEp2zGGeYwhZlF+a+QIbFCyLe4EO6a0wlrPEBn";
        sout << "3DESwdT7ETielqoEQvKdeSQiks0UGvlXRhxKdH+ylQslmdolzE57pMkquAwiFMXddLGFegrctP9s";
        sout << "tmsvLPKWDIqiHy+F79eU6vOfwwS7btaRg5zuRKWkQ+B2CU8F/kx4FR4ZxhK8fzGjMUyjAmHZhEXf";
        sout << "kvnchtB6z0pN7wUf0n+Clxo0DiXlJlRQPo3pZDttbC685azJ3OoH04xS37vxUSx1ir/LWLz/tjkW";
        sout << "iFYq3qxftzK+jU7XzDx2nif7ZLc/+ecfHdQPXK4YZzJ1x8C7SvC7rBLRxnKqTYgv2bL9G1sCU+x6";
        sout << "0hQtMba3x42k//w4RtV2KkazHoMTZc9UuNSsaSoAoGauzw0cs99op7HCpOgoyRu5JeY+fimo2H5C";
        sout << "cXBecQbQdUB0uVxxEQHPwJN7vi94JfbpdnIMLLRjBwRs/2FOmMWbWWcShUYoWDSmJOLaw3Piwtk6";
        sout << "bg/ppKqGAfrzDJkR0n1OZgKvUbnb8WRyZse0W+tO+PcsL1wvwG+8mMJU+AOBs1P/iVLxW/Y4CuXi";
        sout << "/e7SckKJ3vsm/pQawrzhDIjOwofxzBWQ4kODfSEWHZvpQD0HNf/qP6IYfqhUu/0JtRJGLhlQ8hQJ";
        sout << "iJBGtwsCRJWKrBgu6cizrYcA664+XPgjF/FQYLGmPiPrBdrbWjVxSk3tEOgVFOuK+bkI0EX3p0hm";
        sout << "gYbr3oIec4bKzrSgYsIQtHMo1FnQl1xwHL0vH24KF6V6eyYpgVBfg42MNDk/aaCZ4XVIgH0H0wns";
        sout << "sRXftElLUVk8yLhqq9kXBmgHvPZMfA5WTP+KhXFRbfxw0A2nWGbztsniRcoA3N0pGdqwDyOE5VGg";
        sout << "tX94o9eS0eOJzh80SKaHFaX8GtlpVhogNJiMVlzwVoNJASK2Pr7Yp8uIqcUT7+e0VtkdsVlG5wv8";
        sout << "WLEbqmRXrsKLs9f1p23SelLo78kI9nBujEvDSOCChnNwqNPG85kiz24jL0LMaWHAHlnY6uZypDyM";
        sout << "TUmjsyosdrCobZRnQFf4UwUhuNtj3f6sQke+GQhzr474hTpfSDqLGMW6IcE3OcU4x3waC87DPSRC";
        sout << "7PtmJ/+8nWIElEbGJtjS+rL+Ue6faqpkh+dkPC5ZsWHHRvXzyuRNawC15L0kLhhCc9Y+s+fWOppC";
        sout << "iWtPPQKk4PKDA/g5TRA+KPkFH0B6YchdiEaCMLmleDqF9uo+XNzMnHdOKrkTZ29gPosM9w8CpTSF";
        sout << "neDroZT/v1ckXkEZ6rhlVkF2pBmqG0DTL3LPclzO3JC6i6noY+kFU0jSARjXgQU3NXrYpgeRhLuv";
        sout << "hKlC4Fl4xTK8l+p8J8Uk9zKTKsAdyDcAT6rBGfRpmJiN5a+lxuiBSCyygeHDQGhaJFROD53Q/m9M";
        sout << "dxBlTjqMI7r5M2BNFKaZ4vhFdukvCbu2dPm8t10pc4brs7L5TeBVo5qxaFgRbkpS1mTCoDtFH4fi";
        sout << "Wl+zpEdF6VpKrmFaCSSduE2tMhr164re7m4P7CxeJWvYdFfWGD+uDhFM7oPVXvvkC5gYmjdAPIYq";
        sout << "co6IvSMraA9ANQd8b/hO6u+zo+U2Fos9Xe/1u5YWr9JdXZ9oFsdNGQLaLH6VcRrwAyx7+tRjf5Ia";
        sout << "IHli9+TQmZ4tbtxERYe8TaqFqugpCGtvmcE/DFo0BeWgFRFnAY6nyXGgJCrzbxrMdCOBdvIYzuuB";
        sout << "+A68idNO9ifsfNxRalfJCQNymwy4MAylkG8ncZ6Bqx1XztI9ckbD7U7TBMHCWt9xnMxQz9G0HmsQ";
        sout << "pIa0x8tKk5zZ2TyOVe1LjwBXzwhn17i67Ph+NTA2pw8La9KcI2xdlDtAhD+LxRBANo95CeGL9NKp";
        sout << "cDjWMrqwRlvXL0qroKeJuRqtSPGC+hbFEJgTX7iWDq4QJCyvscIm/lWz0ZQIHyXyh3yV/UyGbMXD";
        sout << "hc6mVp20J+AGPR0NCEN3mTh01ON3LJI1t1P6OT8oGM2ofce1YsHLdMlY3uu00ErXy0YF6vz7jft1";
        sout << "0St41Ydx54E5As7cbimxngKnJsFVIdJC8uh8SaAJQJWtQK6DG5sXJVDADSIKXM0TuhRTxRREGQWB";
        sout << "W6Hd2jP7ZArcBCuB2GGMw4sn8iAYMK1LP12hxoZsBZ8iAbohy3MpWZHiE9MDU5PbGRyNsEnuLmQp";
        sout << "APmj5sFUcAsA0MSUYZli2jB2WWAWwTaQ1CGm92tdrdflShh5FR8IhAZrwXPzI/w/1vAianD9yheA";
        sout << "j0cYf/EaB92n9xRSK5zeajIV+DFT/451rNvi0Dqea6cDxRkza31G3d7pWwPkY6WiGdvSzlgy8uJN";
        sout << "pt2gJoZJ6VzzeYD1bsb5X/FYBtYwFuiGicRbUadEA0b736fEC3AG2OZVh5bFmVArBoUukUoBNf8S";
        sout << "gWgzfeYNyL25qa5jaeg+X8okmGtNUzfxLtmJiCY4/A9aDh3yoSCIHwH1we8m18DjMTXYzoc2b99i";
        sout << "18h/UV7FF5xvl5awkZyLjDPPSDtdafb5ufHyNVjblORDSqS2soTzwyoyZm4PTCeiXWaO9Dpz81fM";
        sout << "+bCkqFrXm7qEJqYSrCGLDlxwVZJeHm/lCNpbO+GYq6Cd5CuaJVactLLRre6s43nyD7IxiRfCmb/f";
        sout << "LUyVi5sXEIaWiw80Me64uq/s9ADmuDUkX9Gd8WA+7fyyytSuMpooPNEsVBY9LE5nKWOlOy8Hrqmf";
        sout << "piWc2Pf5nUtyZSsK3p94XbsysthhunMLsiv5j4mcs61xi2IyEgWB5hJ01qk4gQV/8SHPYJ4stRxA";
        sout << "Ea80306xhLLKQjYSpPKHOvoil9kCHIgBzOp6lZas2vOzK/w50AVekKYXFQK2lWMs8TUBzWy5fYPK";
        sout << "CZgcrWP0fShh9pthBw0DBEmpGM6fQZ5XQlBC2hG8HEDbbakqvIUkpuL7jlFde+HW31aTQRluTYl3";
        sout << "huZ21u40SMx86ghv3VUzHpmF5x2b+UIIi+l0yg5FZHXRsth2xRsm4pNO6bspgiL/HrWMfsjwD2Vz";
        sout << "d/2Kx2Dn9FLRUg/ASedunqth0Ovq3z9Qds7pH6QVdBUmtPokcHoC3KKl1gmY7/cN880Az+h0SMpn";
        sout << "eqduvQM9adP2tmuybV5zgKGCt1q6cc0fPPBD1DuwAgr832VjU87nVOl13p4TV9NKX6wvnfRcw1bQ";
        sout << "nJdFr911d2uMjwuPJdKusPo6o86c8YHTOcmUkC2QkMM6gsYp/lK+lv9fwJhvXUKg0aBlNceJ0/eK";
        sout << "LGzHHzsVCweHXjlVY6Z89uMHZZeqp/wzEfatokYF+jIfD9rP+9AyuMIRuIOuXTCkemsGcAHqpg3F";
        sout << "EcSZcaimDAyropuc7CYsVhhxKRDQBYjTbnd0dhIIDZ9WVP/MbG7QRmJF77TB1+a6GlNjoOYEuJfm";
        sout << "RX34p0IQ/ycmc8PcUbFXAC2/epoQKPRprwg2+EbciWSYQe9i8T9gzJVuVHaWF1GjlsNJNvJDnWVW";
        sout << "2ffDvQuZ/YZ//zqKcA6e9A6tTttCUD4XebQmhT5vIesFMuKNUHBvJZwerszeY+AY1Hs8kwTJNMB/";
        sout << "DDj71I1sz1vq7X8OczT4vaHqLDg/4MiyHFatIaGMlbegVLtthaj7BdhwxM7xz0iilKncYQ2zYw9S";
        sout << "wMgYGoTth7eZQe/q0rgzXi25acEvNkbidVbeI+PtUQ1694G/eKRqOYnmaWmhMsCsEUJH5ZI+XhkN";
        sout << "+94T9Tjb6s/P9z0PisH0UAUDT0Rp+DeikJF1h/yLnxhQ3KxIwt9yB+ZlizVXB+6F7xcOAXuVocD+";
        sout << "AyoxOZRmI7dRlFB28ki5Bcl/EHXa70EEFyFao+xc66nv7luVhyscR7PydzdbIlYba2tnkr/QS5RC";
        sout << "kQ+4t8Z2smt7YKo2d/A4Gz3YNk3K3ZbUDWWSClHkcUklQVJDGg4b2da+V9RV8iAuugNuEdDVrs1r";
        sout << "ixGKhyyEGGfIaFUPEaL5/NrAZqBuX6FSuloE7m/MShmkxRuilG5Ngqy9Gb273F5O5x5T6/koSaTc";
        sout << "5tLEdNFmpZYmj7vdIAsHeoyjxmmSycfE8lsCFh1yZRp3I58aXVxBoTHGnQYkQoIeBpr9GBPTo9hx";
        sout << "k05R5M/LAy+Y11NSEW/gRNSiDkmUDGclorU8nz+dLVuyq4ZFX0fGt+IH92B/Ut6oX+S8CaL0iDcf";
        sout << "L+AtFn+m7o5UUKZx9KN2YEv3EbxEdl3m3BsSsgJr/KnVvd88zCEoILj6zZAHE0tkqrYRDX0rKjGc";
        sout << "1LNaJQ+TGXjE0btPlj4hVLWwZHJx9JT70uDDDQ/xz4v8p4Q8MQqfyhf5dc6/GeZtMP031eq1R84J";
        sout << "TeEyGjpF7F4Kngzg1q8mFT8Ay4dmF1rCwvwMl7QGwMAIp0lx+MRC3J9YK5TALpvK2NA70LNDzquo";
        sout << "vuu2cNlAGDtCBJNQ7n9vFEDy37OeXwbTpoJDOWXx4uDL6HQD4yZKxeFbX9AdyS3pTcl12wDkodRU";
        sout << "ESXKNoL9DydL+atZpSrTK06OYqoo4s5ihsGUh/CRJe1owWDoCHJEmT4ghhVeU8YVHxxdVEpOtXw2";
        sout << "csBk3ljCjfpZoXf5yLtEa5Md5JdAP7fVuqDv8sPzg/I0IvXvD24a1RxPcalo5Z5adVfWWGZkC1W/";
        sout << "oBAEgcYFTCVW7IprKK/JuNv1988z19JHhpqMEDNWr7JszAEQ9KRTNtLYjFb/uDCdUSgqiQbV6tjD";
        sout << "PeeKTQbxZ4r6fmtEuV75z/0be62g4t+/aHGNWJjJuZ0P1A6of0LOPZwKhRY2kydC8okBVp3TsM76";
        sout << "9p8yuQAs3WuzaSJR8H2woYQoykHJV/ARapMBuxHlvrhDWFITpyN2LXl5suea3UK1GBJ6HWSiFrIQ";
        sout << "RvMpY13CsRH7uPdx0svXicHK/GRnOPr6ei7cmMsp+nOKXmE15XfatnD8N6OHLImCrfY+bLS1FO4K";
        sout << "EOWti+cmcfz06z70BNUnGSHJqWNohvvGVsre5rimgFSRUJrxN1RTrievQuaVB+hya9rL+dKBRjmf";
        sout << "Uc95nLFFBzuhO/CYEzaGX8JpyyQgh0I38lMww3jK+FRbw3AocP7/rdaNpqi6coY/eFrl8Iv8drWh";
        sout << "5B59c5boqoOa6ZFvIkqB3oJp7ogpO6zFnSS3rGXt0tMWyj5PkSWeN2Tq9pO2gdSM/p0UgN4Ywcpn";
        sout << "rU8gtJgD/zct8G5pH4rAETV3vjfKEnlqG47oIDJzi2PY5zuiSlf3z5pY2nnPjhAFhlBAOGxCV+Ch";
        sout << "i0Y0ziAO3PKo9YXNF8q2hnzroT2o/GXjOZdC56mkRdzYALMv5vkPTQMBqddjahpZDVJLN/jCsnGF";
        sout << "fVOCW57+dzaFVT5Zlsk9xqIaUim+bDg2IHv832FM49MIQx7sJAS4lRmsZNlS1NjWKHwsOtgLPK7+";
        sout << "jRIc6qhU1i7l7cWFd6+oM0U3Sv5yBXTLdTGbWbUniUCn0Izv29BjX9KouaFPRNTyKYfNnoE2dDq/";
        sout << "jNGe+Uxcnbt8vxewpCqRvS7iGBX+Ylf6MW+HkhFu5eKu2pSEK5JyLLS/+kSRHyLhdmhz1PBRh+mr";
        sout << "duozCqvGZN1cMESXzPAVSgKE2sFz7au28raq7YvYI+Pe/8AbD75HPkYlEdVu6SXzwNGrCksJuE1A";
        sout << "u/tAl4GZzEzvyQqUtcf3HD1dWV/ihrtPgXbpCR+GeR9zWrj4MjTDm7ZtsL2NDm599UTNNPJgaIxD";
        sout << "5coKin30t2hOg6LzFoGKpAwGTijauINY/xAqgQBA8vEQ7uYGK2bkPn9llAAG9e2L+KPKVS6nLyFf";
        sout << "unzr/rPkxU4VITFN6V9GGZoJQZ/QFiCm6kLO+beLPgsPkZAqJNO5Gl9OTa6728Ew05ZnziMsWJaM";
        sout << "OaAqjBrE92wtFITs3Qdr+CHKgKqsrGXEUK6hfylB0pOhtNq78gtwsQ8rFzwyu3hoDv4YVEt6FBOx";
        sout << "zb1KBFOMz2x/RZ1qTdO9bMONUe31rjsOTmiFu8/CQyYfjciF8cp367jbzgcWV2aFhQkY2tA7SL/P";
        sout << "HCU5bSb4qpqJG4zg+RPv2Hx/6DpmIDXwJgSajh3a7O+HfcMKIRvWOXqKkc4LPK1RmMy8aDYzNnav";
        sout << "frd4Ii8a2KCeVqsGmJylpypEeMyjQCX8CcIYBYOEVQmUQGAoO71Cauftv1pt1yFAUzpDn+gGGSmN";
        sout << "pp2ZCm/qPb63lb6Kz94Piq6oVw23zqFrr0pqXr2TEi4e7jTLzMGCcXBUj/qNiCly7TmlFzM4obpO";
        sout << "1ev0yo6ccmAF8H4BOeyX7lqeqlwZHmpjc/8oa7QwuQnBXB7c7HHm+L9F3N9QoPnEqLtSmrJmNPoL";
        sout << "qk1d0l1174BQPBZCl6dcCHQdefAZAQ5v66WcPAoZNlt0lVL6CBman1pk5p0e4zU1EPrqYIUxzfBG";
        sout << "mLv2zWim3OpVjpYpB82fhtIlyIwOst+2rkbeCdIm/3X54LiC/hudzp7zUa4pe99DM8jenauzTIR6";
        sout << "BdqYbHBRQTc5rKaUmRDq/+JPVaG2dAjWVTdPHLs+rFM3MvLdd0wPG2T26uwwQyhAx+PHT9JEhU+t";
        sout << "pSJpE/s6LlZmqt6RPuGgYuO6jifhECGWmdy6SgT2wYl/9REvPSsMIMiB9DAbdKAFv7ios1KtcjOq";
        sout << "pIOUs4NKeJ3QMSU3lE5JXf0V45VBkJ5JfO9lyCMgHRGFd89mRf8/HON8GYkSidyFF+d2Z+po7tHS";
        sout << "Bhfq86T6T4vTUDE3KCIcsir6kE+hyZylWw+fnBRzWYVYMBp2YCKHybxBKdkxpvAnDLibZYdyEtd/";
        sout << "20RPeXxxk54lJkFAjNy6vtnh6vLomfNALcZ8oqS8iZWX5v4q35b152XHo9lEWbTxokbbXeMmqLdL";
        sout << "UjBrCPkD1j9ogboDfWD5TNg60dJVikPCyUHbTSTTEU+I9niREiVDdZToXbeaeRgOKYxtnwWiA/FM";
        sout << "BYXiPb57y+/il8TD1ZT74JRjz6kAmSa3bM7GClr//V3Mdl3TpoP949ZhX1L16IHc8WwgE0rQnQHV";
        sout << "NxPxVWS0EhAdmlYB73ib5jPua1rtlwRbeYbDF2iNsuw8ss8phioK+ZR6BnZX0XZN2Tw58Fa1e5kV";
        sout << "y2kHZGIkY6hk/lrA/vAP+QuV8Pxb/P7VXVsgkHglBYuZkkZLgjqAZDvPjvZcBqEDQjCHrG6V4woD";
        sout << "X7xWGVYXVRt5vJcwOX0hKCcUSv3+1XL6+ID0urNTuFqMw96QKtb3//H6qZvZIO/bYGjJcDVbQHJB";
        sout << "6Q//OjqKgSiw67SotBAvIVQrgDeHUzaS7JdjyvWHCooClKrAiIVcnKX02r1y+mVpeL36166MC9D7";
        sout << "dAi/coCiOwcQ6STbBY2PPhWrko31E8o/l3uDeC4cl3TIVjkFjS61GMogqvd2ISZ3Egq1jZkRYMln";
        sout << "962j1y32UizRikxEK+2d0UREZ70oZzW3UoD6YqshoElP78GkH7HGS8hjpL+UhuCjCura1FK4/qVG";
        sout << "yp5GneObh4S9DzJV+IZuviLd6drknWn+nKYS4YPWkbOqBu7RBvh3eb4bQMzvVl2thL1b4Ff8Q79d";
        sout << "mWxu8ajtptUs0OrSor9gnqrLu3K3RWRTWSElrrMHjCD5GZcsrR/qp7Ip4GDBRPCVJNLa1Tmm8TMc";
        sout << "kTkGtbN0E96kT2qVA4/s0vYCxntAUPunP0k/JtWg5YzVXQi5/hKDdIE3EfpV2PXjveDoBasXAmEi";
        sout << "6oUcZk1zKqwFygj1793MM02voesHS7BxWz50W6yD+SozDsnliV7fY4S9+8vj8CipRYSD8t7MoczJ";
        sout << "hg2tEMYVVBo/nV/4l607tvicVT8HYWOkBhuOgeMNrocTp/3zcfiCODeda8rzbGPWjmKOCPcgOmTD";
        sout << "TO5jx0f568jqAuhFgeg6wv/3uXuY3mmFGWsCx4Sf+2nlWqhCMKmSsRT84UXcFYpp0Kcvx1OUFC/c";
        sout << "E64GAqCYYhzb3hF7jys36qQefbfm9+t3owVhP0d9udWNBzw/QeMJ6djmHvk11Tl3qvjGUGV0iXh6";
        sout << "4ZxyCeLQWZA6cNcN8Ovd2vRrtQ8SHlFPhMKqoevmMLkyZKMrD9CUPHmgzlpTuAasa7PEnbaAFHcO";
        sout << "sVE1KNm7uMU7QjFI8u10VRJ6qBpTx6Z3GXPq7Jslk2V6z0xmH/elkzMAu2Wr3MId0/7GCuheVZhh";
        sout << "VAf9EWJ2ZfZxGBOXd8Io9eiatd/VdlOh7FBglIGSpx8UHU3jzpSu1fcnCjVRg3XxWmUI0iRrqxQc";
        sout << "iVT6ttiezDImpPlHxP2OjgIghWD6EZ2Gesunu93a2lep85rEZqN8sACV2sXDy5K7CySNLyhNE7fX";
        sout << "eV1bU3FdOstad/82yh7TJdIpIdFV1tEOV+gAOMZa+516EjdnDs4WJpfWHPWG6xdVJWAvCHss1B8s";
        sout << "k11txpDa6vs3+NCx/mF/6ElB7TxkisPo1m+KfgjBGpI/YHlm6c216WwSh1k1hLk0T3s4wFEb8M5w";
        sout << "BlbxPkt89Y0wWwc/Eg37XquMIme6aBZjU3CZK1NwiAPkKXa9Y6fBTZWLT3W/NpP2Vk8KGxae2fRo";
        sout << "F83VOgFxb1SUIePgZ2vMS/6OnuRxNqiwkcDEI15uVcK+l1AynHFqODaA31sxqQuHnw/FOrPG5yJR";
        sout << "OvmgJOJ9ss9QZkvXaTZFc3vfEZElcKdW9K5xEZfiymZWX9Qihiv4PG/L+qo5Tm6cFxi/8MVW7Tgd";
        sout << "OV5whxO+RWOCj1kyuILneXxrwiYL3tn74Z4+tT6oP2I9l2UXF574JXrOzLLeBhULrD9fpPFlb/lM";
        sout << "5YfP97MRad6MEZfY5uMUa36kUR8s7FMdTpSyKqhEmzmepGD7JI0uGPNutfSDO5SJsPfK8Yh1uPWA";
        sout << "5J5d/NEUp/fqvoniWgm/2ye0ApW90EtX9eDE/DyfPQimpqOdFBO7/EvDkDHlZ0u08F7+sCavgzxE";
        sout << "jHMXJ+uOF5q96vNnGDhyc8WC9NRzX66GChvTCh7nDhBYLMCZzSvWU7wPWC9OLOCvA+lyTTGvFCgs";
        sout << "qCrk5Hoc5etEIrpyOCe6evI681jAoNI0KK4Opb/vqVtKgLTxJhBbi/EVhJTzdALEB/WduuYpfqqD";
        sout << "sUbdDlNRdSGgricRZUxD0iN5GZUBDEzVzp6moEW+Q7NxsryW6/88Ow3ky1fm41QPwQ40Rk45cnIU";
        sout << "5nZxTrYUkBoaWR/Lh43XtKDh3cBhADGiaMIPBVJL/tOUtXTCj93ca73/LqanFOO/rrnROhXUCn1h";
        sout << "+LWEWn553i9PF0CqFkFjtK/dP68wPQ7y09/NYYw9P86YdgoWwm6ZlqskD2ByyYgR5KimmyJLQyxb";
        sout << "CmITcgVbucDiDxUUFG4Rq9Sgn4PW6jGaoujCgclaxOgLnM7lqREfW8XlN/yH8Pvyt7hjAIWNOHyg";
        sout << "hiscgSb1ev+btDSm+aRCjPgRNfBDApl2zINR7ey0wA+wOl6wZh7cxYwbeb55/UqH6OzEXbjJzjki";
        sout << "6c9Z95PGsu8U+FaBJmWyJWqb/SxBxsf4zPH154KF7wE6b0AoP2Y1tGD2hDN8WdEthVWbztLm66nI";
        sout << "n0fz38pv7CNJCuVf+jeqvk6bh8/dI0hVF0CsMd5uVe+fOLzK3nmIma5nB/TQvcCqfR1YjDGV3mhd";
        sout << "QPQ1a7ypy6WplKOU5LamJw8IaYrik3Gg2tLo63FfE8njUkrYdtqiYpgk+okF8vF4S8c8NdWlgTNn";
        sout << "KE+9+dPQZr174KhRz1JwzspIUqW/d/QjGIgJm5NcXAo07/hgpfI2zfHlkGP5ETxLnquzkCcCsrnK";
        sout << "IQ+vacYRwHRVSTVWpvQiDXaNK7RJrfW2ei23/bW8YkE72+lZIACjehXSDgKiBg0EIStpOVLwTSln";
        sout << "fXVW5fHQkyAc4fOVX6bGqcBu40hHN/LRFJY7Jv5qS/AjCtLUR4UCRhMWAo7ITDLgbXoIDKnyrRLF";
        sout << "vS7LGaJ9IHbFv0Lw8Xiqr1YSpjnLZZDcREIsjCf+2G/7kNBY96DPmp63sGuBfIR0f/I6MzUi8HNG";
        sout << "R+EBmcCO0AfPzTeoGkIM9+5YmMpXwvMU0zCz71PxeLlnTUVUggnXTBAa92YS6HuhGKIl5aU0OCXW";
        sout << "76CfhWijfNZCG33EdpOJbXxbtbX97y+Xjn1KyPnNW/Wyk5VO8lqD2ZGC1Wnohpx4q2VtDkVzw9h+";
        sout << "DDbj2hpyfv2byvYJPUSlZ4bJ77aCkYuZsjMOaz4MYjVfgdi7svHdnIBauBgv+rubJAwx8oB9bk3K";
        sout << "Wy2EDPfLHGxtUKt9+keHGnx9xZ9vGjLXzaLiGltRQ5YDiVK/vXPUc8eKy7p+NyNu80yopPIjJcgV";
        sout << "IC0mPwR2BvdGVweS/iLAuMVJw3STNQyl7XVej6lppzt2SWidtjqxYnYjJwRYNnFr0l7SgVA9Tink";
        sout << "jZz7M4OGdDN521PxFXjOcRRRM1Kr7/56n+VF+LdTRCihmUsOuGi4bBlZ+eKTcUmg7FX6LXKptWLm";
        sout << "h6QUA5ZXcUZ9XUrJb1AxaHmfP/XXd9r1bsi+F3JdFUPuCOZHrzok6cshx+9r8WH1MNCBBVTeUrFl";
        sout << "+65D1RufbIVoqZfQ8l3+zghe+U/ujnqEox8ysQAZkhYG6beS6ksj/QhWbGno9W8TeaXr2NvcF2ct";
        sout << "PAh85hlFNWcxBdGnHXl8fQJ+p5+t8+jftQLoVqd6B9/beAUgp9KYrLugImLUnzw6Q7PjXCiFmo8d";
        sout << "gqPvKtiqj4mKu5L5L4/ul2zdN4zQtRE6tm24+ENVyfJ00adWwqEGxrCIvUMiNbOzSI9TtoTVrKgx";
        sout << "lON3nzfcVy3ucT93AtQfa2reu/HwRVNhn0GP/xZGBCy746if7jr5Fa34dAESoJ1mlelIKOion8QJ";
        sout << "bsgQixF1UWwd1/O4DgHG+U43HYhLWOSw+NOnajpEaVoKB+jf/M0u5+BQvbwCcE6zd+Vhmhzy6x7i";
        sout << "5ySbdBren80S4wUt97VFIq7HI0KvvWpkbAeQjGOhAWIqRrtbJup9yQa09VJiR0hiMQ8K2UlTx3By";
        sout << "BFtRAiJO5mn85Q0YxqyogwSE2Kf4F7qu3EBnT/rEta+a0e/UcElFcZPpJfqSwS18mNBPDEyagvBv";
        sout << "BQiEumckYIVwtFMpfrvELupxCTY51anDR44j+RTJlQMf91CHDZi9eewt9Rfo4ja5HTTdK5Wmjkvt";
        sout << "DR2SBS/Z4YJX4elAbie5bsrFWR7IH9M8M/vtR++6Sw2SKEAQjS7D23xmoKXD/bC9JWoz9P0yICkJ";
        sout << "Ckdwh6H+FNzh6Ms8qlcEWtQD28SI271I9tt0Yiw7FvwIYH/+aG7+PnXpyei7Y8OmV2scS2CaAR2e";
        sout << "+k0qUumoXI/yPunLeJE8BQHjslaw8eglGnb7YjfgmoGPC3lzrc69UX3Okmnb8JJUoC3KEsN0Zf0n";
        sout << "GPErNUSm7ABFRAQvHvTiDxfGyf8aN6UNYZDC0rNO5MgYpQxvmeTnCMtXz0P9OEpaBs3ZuNoJVR+n";
        sout << "s1zhiz5JteC9VmCFM8PknTz2tQXBWonqXdHxCPOIi7k0K1Jn8ddygrfUczXeMeZY/H/akoWADYHT";
        sout << "cGPqpi/81wQUaE4GmV6X27XcuHUcELZTkyTqpkBprMoPJJDspnbFj0aaHiz19Ws0weE3wWhUFiA4";
        sout << "D4WlUi4uF40OEAhnv0qcSVHil71C7AEpTcS1dxe+GSs02PppfT+PZDexeWk/S/C8E+O/Fr4BVfO0";
        sout << "G+M+XdcvagPHLZYLTEJhLhvQzPQRbOb9vdPUhiIsodTUzejdQ+zA6k7ynQqNEDAaVwyy82AQRyoi";
        sout << "4BFGQzQ2hYCUp+In9B/69qyBSvT+q6E1YDUARctcr3h/AR/3J7uG4VdxhFA6YOKAnJGCSCMdunwi";
        sout << "hb1or9pPHhW5rDfvmtH9iZCsuvvJmFA8FXmAURRDRziSxYR/zcApR0JBT63mtyw3lMCHP1qDg95w";
        sout << "0ysG3hLdyMI3lnM/g1W0tg1hvXR0C3qjivvuthJHyt3fqSZbVnvNEkD2k+7BoG6KWk5EpoSqv+Ni";
        sout << "znQvLFggfPOtd6Y8BkmBcNhCTCcyMa2ZkvzUNO7419VR/gNeU56athpUvD21r8d0/MWrEi2yKWsM";
        sout << "aCVwYr/eAgp9kpn9i8RuSJmxe4NTb7Isqn25vOfUeUY17vS+c6wjCDqbkmukld4nOHl8uHlHdJoK";
        sout << "elhC/yXSgUyOM8yKFSyw+Ap6v9CISYh3y09g6gBL35LjS7Vb4Ew/ks2t2L9zWrU+IC4BLcQDmMfG";
        sout << "QUk5tt1n05bmuLmnSboYhPj2MNT9/EiqHY9AfwPhMYmWbcSCshdShSoYs1aciC1gc1H393KbvaTl";
        sout << "N6TeDKz6bEkvM0mOu+wRDR5u+tnr0wWSW8JkMt+WkjPvqWkGYiKZuhJT/4rTPRhJEKXE34gxeyFA";
        sout << "16oufb8W9C8Fj30WPJq1bd43y/r0Fs1dwOyZAyLsHv/2AtO5/Jw2KEHsTIgolGr0Qe6//jc/c50R";
        sout << "TN/aDVTBCbvhcRtV/zOuy7oqb+p0IM3p7eWtKFrZ1wXly7FvvPQ5ODn9CnG5wo59XTikanWOMLKf";
        sout << "MwQt89ZW60v9bQTgIUnFYpVlp9SF2XPxmeW6x+NVCEOiXZQ8AM3nDiLA5M0ctrI2DzzLeHoEKWem";
        sout << "TvVZ1MrwJ8jZoRsigx45HqoB353+bXlS+5vMB/M+zUEPu/HHuU5k8zqwU93NFkTR208ZtecG3IPs";
        sout << "ENQd7J10XsbhbYhAvEkU9ZS/3FS7aK7bDvf7cqD/QcGCfIad4rk1Ks6nuGSeR7hUrH1NK1fe9lpY";
        sout << "lNk3EaNPlBEBmNkbGpNIUcJ7ntUy+b1q+VoC+q300a4qo3gIOkN3s25NaDJime/eJmlRYZhH4ip8";
        sout << "6+m+nA8as1/T0/4d7HXiFWQwZ49NZsSESry1yCs97C5IQ6nScahxHDi72AbQeNDB+RtGaQJiOUi7";
        sout << "NSBNuSlm9G9GpR55HvMi/JUqF09BrOn+49zwYqMbSkf8CjoenLM7UzCoywGlk0nXSBsbANAz7D2B";
        sout << "65qWXswtvr4xnCSfRLUNHs3AtlfEqpsdlaF9gDQm5Z4IhT1WOXbETcY5S8T/5DDBIoWi9rHnKuxM";
        sout << "+zmu881jhj3d9p8fbcH65hevrYM49+ZQfWPXTMUY77YbwYTGmYgScAWmqaMugoMWD1ocpYYRM0IJ";
        sout << "+SEiUb57moAOoEeiYZcPqmckTWuHJhYtgbuBojwXqaK/qvDssM/59aTMWagOYHcapC4gBG+s99No";
        sout << "pOCnbe2brIm4+6xWs7LzSA38RZHZSdh66V3n+83R0/wAIw9+X35SXMwrXC96OqXF/6AFvqkL2Wnk";
        sout << "SBbvyq0txWR6b7AaZ418Dmngg3yQh04fwc8xZLy7/1ZYAbGLRRV1mNrpc2Fa1kLjxoRHZMBA75Pt";
        sout << "HirY4CHOvKaEdlk27BW2px1QCTCkZQ/gojWhiZ1kPUAUiW7VcyFSzjtXzswHEIAnGR2dWHgGZDVT";
        sout << "OVuBJ0nTPs8itQ2Htelag60Et9jrwDZzYa4Rhy1FjngWN1S/QAp9iGe95SRVXuBtLNgAVp+sx7SU";
        sout << "VOECSHoLfpSeZPvlm5ibeSN83gFbIG2rsTZ3IlvJjWq82Npzas6p9WVKTEPGS+Ux8nWIBT/enw7o";
        sout << "7KX9phVWQqcYH5IB2waRO+Ke7h6/y696NQMq0R4Xbki9lmjoWNKFtM+GgLygVqxWWWp9iyQFkUQx";
        sout << "7tRJT1da0ImlgCXS/uTTRxvcG9d/E5FMotFa6mA7py7P+eraScFdEHL4J0kA";

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





