# nmler
package to analyze directionality and branching patterns of skeletons made from tracing EM/XNH data on WebKnossos

## Code

**nmler/core.py** 
  - Functionality to:
    - Remap tracing coordinates from bin2 to bin1 space
    - Create a Navis Neuron List from a single tracing sequence
    - Calculate first principal component of a skeleton (tracing sequence) and estimate best fit line
