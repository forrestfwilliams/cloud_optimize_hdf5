# Experimental Plan

Have four variables to play with (in initial importance order):
1. PAGE buffer size
2. Data Chunk size
3. Page Cache size
4. Data Cache size

## Plan
For each group (page or chunk), run separate experiments. Each experiment will run five times in a row (to make sure the data is hot), and the minimum time will be recorded.

### Page Size
pick five reasonable values for each 
page size: [1, 2, 4, 8, 16]
cache size: [2, 8, 32, 128, 512]

combine these into 25 unique combinations
read only the first data cell

### Chunk Size
pick five reasonable values for each 
chunk size: [4, 8, 16, 24, 48]
cache size: [4, 16, 64, 256, 1024]

combine these into 25 unique combinations
read a quarter of the data
