naive: naive.cpp algorithms/Naive.hpp algorithms/Reference.hpp Utils.hpp 
	icpx -fsycl naive.cpp -o naive 

.PHONY: clean

clean:
	rm -f naive *.o 