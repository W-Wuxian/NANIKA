program FactorialTest
    implicit none
    integer :: num, result
    integer :: calculate_factorial
    ! Set the input value for which we want to find the factorial
    num = 5
    
    ! Calculate and print the factorial of 'num'
    result = calculate_factorial(num)
    print *, "Factorial of", num, "is", result
end program FactorialTest

! Recursive function to compute factorial
recursive function calculate_factorial(n) result(res_val)
    implicit none
    integer, intent(in) :: n
    integer :: res_val
    
    if (n == 1 .or. n == 0) then
        res_val = 1
    else
        res_val = n * calculate_factorial(n - 1)
    endif
end function calculate_factorial
