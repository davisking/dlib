
/*

    This file tests all the ways of using jvector and jvector_crit. 

*/


import net.dlib.*;

public class swig_test
{
    public static int sum(long[] arr)
    {
        int s = 0;
        for (int i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }
    public static void zero(long[] arr)
    {
        for (int i = 0; i < arr.length; ++i)
            arr[i] = 0;
    }

    public static int sum(byte[] arr)
    {
        int s = 0;
        for (int i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }
    public static void zero(byte[] arr)
    {
        for (int i = 0; i < arr.length; ++i)
            arr[i] = 0;
    }
    public static int sum(short[] arr)
    {
        int s = 0;
        for (int i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }
    public static void zero(short[] arr)
    {
        for (int i = 0; i < arr.length; ++i)
            arr[i] = 0;
    }

    public static int sum(int[] arr)
    {
        int s = 0;
        for (int i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }
    public static void zero(int[] arr)
    {
        for (int i = 0; i < arr.length; ++i)
            arr[i] = 0;
    }

    public static void assertIs28(int val)
    {
        if (val != 28)
        {
            throw new RuntimeException("Test failed " + val);
        }
    }

    public static void assertIsEqual(int val1, int val2)
    {
        if (val1 != val2)
        {
            throw new RuntimeException("Test failed " + val1 + " should be equal to " + val2);
        }
    }

    public static double sum(double[] arr)
    {
        double s = 0;
        for (int i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }
    public static void zero(double[] arr)
    {
        for (int i = 0; i < arr.length; ++i)
            arr[i] = 0;
    }

    public static void assertIs28(double val)
    {
        if (val != 28)
        {
            throw new RuntimeException("Test failed " + val);
        }
    }

    public static float sum(float[] arr)
    {
        float s = 0;
        for (int i = 0; i < arr.length; ++i)
            s += arr[i];
        return s;
    }
    public static void zero(float[] arr)
    {
        for (int i = 0; i < arr.length; ++i)
            arr[i] = 0;
    }

    public static void assertIs28(float val)
    {
        if (val != 28)
        {
            throw new RuntimeException("Test failed " + val);
        }
    }

    public static void main(String[] args)
    {
        {
            float[] arr = new float[8];

            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
            }
            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum_crit(arr));
            }
        }
        {
            double[] arr = new double[8];

            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
            }
            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum_crit(arr));
            }
        }
        {
            byte[] arr = new byte[8];

            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
            }
            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum_crit(arr));
            }
        }
        {
            long[] arr = new long[8];

            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
            }
            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum_crit(arr));
            }
        }
        {
            short[] arr = new short[8];

            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
            }
            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum_crit(arr));
            }
        }
        {
            int[] arr = new int[8];

            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
            }
            for (int round = 0; round < 100; ++round)
            {
                zero(arr); global.assign(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum(arr));
                zero(arr); global.assign_crit(arr);
                assertIs28(sum(arr));
                assertIs28(global.sum_crit(arr));
            }
        }
        {
            int[] a = global.make_an_array(4);
            for (int i = 0; i < a.length; ++i)
            {
                assertIsEqual(a[i], i);
            }
        }

        System.out.println("\n\n   ALL TESTS COMPLETED SUCCESSFULLY\n");
    }
}
