# attack the bot server with lots of requests

import requests
import threading
import time
import random
import string
import json


exceptions = {
	"signup": 0,
	"login": 0
}

failed = {
	"signup": 0,
	"login": 0
}

def random_string(length):
	letters = string.ascii_lowercase
	return ''.join(random.choice(letters) for _ in range(length))

def attack():
	for _ in range(10000):
		username = random_string(10)
		password = random_string(10)
		data = {
			"signup": '',
			"username": username,
			"password": password
		}
		try:
			r = requests.post("http://localhost:45454/do_signup", files=data)
			if r.status_code != 200:
				failed["signup"] += 1
		except Exception:
			exceptions["signup"] += 1

		# print(r.text)

def attack2():
	for _ in range(10000):
		username = random_string(10)
		password = random_string(10)
		data = {
			"login": '',
			"username": username,
			"password": password
		}
		try:
			r = requests.post("http://localhost:45454/do_login", files=data)
			if r.status_code != 200:
				failed["login"] += 1
		except Exception:
			exceptions["login"] += 1

t = time.time()
# run the attack
attack()
attack2()
tt = time.time() - t
# print the results
print("Exceptions:")
print(exceptions)
print("Failed:")
print(failed)

print(f"Time taken: {str(tt)}")



exceptions = {
	"signup": 0,
	"login": 0
}

failed = {
	"signup": 0,
	"login": 0
}

# run threaded attack
t1 = threading.Thread(target=attack)
t2 = threading.Thread(target=attack2)

t = time.time()

t1.start()
t2.start()
t1.join()
t2.join()

tt = time.time() - t

# print the results
print("Exceptions:")
print(exceptions)
print("Failed:")
print(failed)

print(f"Time taken: {str(tt)}")