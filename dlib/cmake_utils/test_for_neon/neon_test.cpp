#ifdef __ARM_NEON__
#warning "__ARM_NEON__ is defined."
int foo(int a, int b)
{

    return a+b;
}
#else
#warning "__ARM_NEON__ not defined."
#error "No NEON!"
int foo(int a, int b)
{

    return a+b;
}


#endif

int main()
{
	return 0;
}

// ------------------------------------------------------------------------------------

