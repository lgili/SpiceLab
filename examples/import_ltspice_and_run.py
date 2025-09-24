from spicelab.analysis import TRAN
from spicelab.io.ltspice_parser import from_ltspice_file

c = from_ltspice_file("my_filter.cir")
res = TRAN("1us", "2ms").run(c)
print("traces:", res.traces.names)
