naive: naive.cpp algorithms/Naive.hpp algorithms/Reference.hpp Utils.hpp 
	icpx -fsycl naive.cpp -o naive 

tiled: naive.cpp algorithms/Tiled.hpp algorithms/Reference.hpp Utils.hpp 
	icpx -fsycl tiled.cpp -o tiled 

.PHONY: clean

clean:
	rm -f naive tiled *.o 