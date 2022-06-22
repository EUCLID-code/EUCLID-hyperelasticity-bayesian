import os
import time
import sys
from datetime import datetime
import warnings

def show_current_time():
	print('\n--------------------------------')
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	print("Current Time =", current_time)
	print('--------------------------------\n')

def progressbar(it, prefix="", show_progress_bar=True, size=50, file=sys.stdout):
	count = len(it)
	def show(j):
		if(show_progress_bar):
			x = int(size*j/count)
			file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
			file.flush()
	show(0)
	for i, item in enumerate(it):
		yield item
		show(i+1)
	file.write("\n")
	file.flush()
