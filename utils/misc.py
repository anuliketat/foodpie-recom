import datetime
import traceback
from operator import itemgetter


def sort_tuple(data, sort_key, descending=True):
	return sorted(data, key=itemgetter(sort_key), reverse=descending)

def get_traceback(e):
    tb = (
			"Traceback (most recent call last):\n"
			+ "".join(
			    traceback.format_list(
			        traceback.extract_tb(
			            e.__traceback__
		            )
		        )
		    )
			+ type(e).__name__
			+ ": "
			+ str(e)
    	)
    return tb

def logger(product, event_type, event):
	text = 'UTC: {} | {} | {} | {}'.format(datetime.datetime.utcnow(), str(product), str(event_type), str(event))
	print(text)
