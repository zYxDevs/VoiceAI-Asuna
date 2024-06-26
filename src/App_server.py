#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint:disable=C0413

__version__ = "0.3"

import datetime
import os
import sys
import shutil
import tempfile
from typing import Tuple, Union
# import time
from http import HTTPStatus
import json
# import traceback
import importlib
import re
from http.cookies import SimpleCookie

# SELFMADE LIBS

from CONFIG import appConfig
from user_handler import User, user_handler
from OS_sys import check_internet
from print_text3 import xprint
from REGEX_TOOLS import web_re

# PYROBOX SERVER MODULES

from pyroboxCore import SimpleHTTPRequestHandler as SH
from pyroboxCore import runner as pyrobox_runner
from pyroboxCore import config as pyrobox_config
from pyroboxCore import DealPostData as DPD
from pyroboxCore import PostError


# CHAT LIB

import Chat_raw2
from CHAT_TOOLS import for_voice
from voice_4_live2d import get_audio
from msg_class import MessageObj

pyrobox_config.log_location = appConfig.log_location
true = T = True
false = F = False
null = N = None





class Tools:
	def __init__(self):
		self.styles = {
			"equal" : "=",
			"star"    : "*",
			"hash"  : "#",
			"dash"  : "-",
			"udash": "_"
		}

	def text_box(self, *text, style = "equal"):
		"""
		Returns a string of text with a border around it.
		"""
		text = " ".join(map(str, text))
		term_col = shutil.get_terminal_size()[0]

		s = self.styles[style] if style in self.styles else style
		tt = ""
		for i in text.split('\n'):
			tt += i.center(term_col) + '\n'
		return f"\n\n{s*term_col}\n{tt}{s*term_col}\n\n"



tools = Tools()


def join_path(*paths:str):
	"""join multiple path parts, same as `os.path.join()`
	"""
	return os.path.join(*paths)

#############################################
#             MESSAGE HANDLER               #
#############################################




#############################################
#             SERVER HANDLER                #
#############################################

######### Exception Handling #########

# class 

######### UPDATE CORS POLICY #########

SH.allow_CORS("GET", '*')

######### HANDLE GET REQUEST #########

def handle_user_cookie(self: SH, on_ok="/", on_fail="/login") -> Tuple[Union[User, int], Union[str, int]]:
	"""
		Handle user cookie
		- if user is logged in, (if on_ok: redirect to on_ok n return 0,0) else return user, uid
		- if user is not logged in, (if on_fail: redirect to on_fail n return 0,0) else return -1,-1
	"""
	cookie = self.cookie
	#print(cookie)
	def get(k):
		x = cookie.get(k)
		if x is not None:
			return x.value
		return ""
	username = get("uname")
	uid = get("uid")

	user = None
	validity = auth_uname_pass_data(username, uid, 40)
	if validity == True:
		user = user_handler.server_verify(username, uid)

	# print([user, validity])
	if not (user and validity):
		if on_fail:
			self.redirect(on_fail)
			return 0, 0

	else:
		if on_ok:
			self.redirect(on_ok)
			return 0, 0
			
		return user, uid

	return -1, -1

@SH.on_req('GET', '/favicon.ico')
def send_favico(self: SH, *args, **kwargs):
	"""
	re-direct favicon.ico request to cloud to make server less file bloated
	"""
	self.redirect('/icons/icon-512x512.png')

	return

@SH.on_req('GET', url_regex='/@fonts/.*')
def send_font(self: SH, *args, **kwargs):
	"""
	re-direct font request to specific directory
	"""

	file = web_re.gen_link_facts(self.path)["path"].split("/")[-1]
	
	if not os.path.exists(pyrobox_config.ftp_dir+"/fonts/"+file):
		self.send_error(HTTPStatus.NOT_FOUND, "File not found")
		return None

	return self.send_file(pyrobox_config.ftp_dir+"/fonts/"+file)

@SH.on_req('GET', url_regex='/@scripts/.*')
def send_font(self: SH, *args, **kwargs):
	"""
	re-direct scripts request to specific directory
	"""

	file = web_re.gen_link_facts(self.path)["path"].split("/")[-1]
	
	if not os.path.exists(pyrobox_config.ftp_dir+"/scripts/"+file):
		self.send_error(HTTPStatus.NOT_FOUND, "File not found")
		return None


	if file.endswith(".css"):
		with open(pyrobox_config.ftp_dir+"/scripts/"+file, "rb") as f:
			file_data =	f.read()
		return self.send_css(file_data)

	return self.send_file(pyrobox_config.ftp_dir+"/scripts/"+file)



@SH.on_req('GET', '/')
def send_homepage(self: SH, *args, **kwargs):
	"""
	returns the main page as home
	"""
	user, uid = handle_user_cookie(self, on_ok="", on_fail="/signup")
	#print(cookie_check)
	if (user, uid) == (0, 0):
		return None


	return self.send_file(join_path(pyrobox_config.ftp_dir, "html_page.html"), cache_control="no-store")

@SH.on_req('GET', '/login')
def send_login(self: SH, *args, **kwargs):
	"""
	returns login.html on login request
	js will redirect here or to home based on wheather user is logged in or not
	"""
	user, uid  = handle_user_cookie(self, on_fail="")

	if (user, uid) == (0, 0):
		return None

	return self.send_file(join_path(pyrobox_config.ftp_dir, "html_login.html"), cache_control="no-store")

@SH.on_req('GET', '/signup')
def send_signup(self: SH, *args, **kwargs):
	"""
	returns signup.html on signup request
	js will redirect here or to home based on wheather user is logged in or not
	"""
	user, uid  = handle_user_cookie(self, on_fail="")
	if (user, uid) == (0, 0):
		return None

	return self.send_file(join_path(pyrobox_config.ftp_dir, "html_signup.html"), cache_control="no-store")




@SH.on_req('GET', '/test')
def send_test_page(self: SH, *args, **kwargs):
	"""
	returns signup.html on signup request
	js will redirect here or to home based on wheather user is logged in or not
	"""
	user, uid  = handle_user_cookie(self, on_ok="/", on_fail=None)
	if (user, uid) == (0, 0):
		return None



	user_handler.server_signup("Test_user", "TEST")

	# ACCESS THE USER
	user = user_handler.get_user("Test_user")
	# print(user)
	cookie = add_user_cookie(user.username, user.id)

	self.send_response(200)
	self.send_header_string(cookie.output())

	return self.send_file(join_path(pyrobox_config.ftp_dir, "html_page.html"), cache_control="no-store")

	#return self.send_file(join_path(pyrobox_config.ftp_dir, "html_signup.html"), cache_control="no-store")

@SH.on_req('GET', '/dl_data')
def send_dl_data(self: SH, *args, **kwargs):
	user, uid  = handle_user_cookie(self, on_fail="/signup", on_ok="")
	if (user, uid) == (0, 0):
		return None

	if not appConfig.allow_dl_data:
		return self.send_error(HTTPStatus.NOT_FOUND)

	# use shutil zip to compress data folder and send
	temp = tempfile.NamedTemporaryFile()
	YYYY_MM_DD_HH_MM = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
	file_name = f"Asuna_Data_{YYYY_MM_DD_HH_MM}.zip"

	file = shutil.make_archive(temp.name, 'zip', appConfig.main_data_dir)

	return self.send_file(
		file, 
		cache_control="no-store", 
		download=True, 
		filename=file_name
	)


@SH.on_req('GET', '/voice')
def send_voice(self: SH, *args, **kwargs):
	user, uid = handle_user_cookie(self, on_fail="/signup", on_ok="")
	if (user, uid) == (0, 0):
		return None

	# print(self.query)
	voice = self.query.get("id", None)
	if voice is None:
		return None

	voice = voice[0]

	path = join_path(appConfig.temp_file, "audio/", voice)
	# print(path)


	return self.send_file(path)

@SH.on_req('GET')
def send_default(self: SH, *args, **kwargs):
	"""
	Serves as default GET request handler and returns files on file system under the hosted directory
	"""
	path = kwargs.get('path', '')
	url_path = kwargs.get('url_path', '')
	spathsplit = kwargs.get('spathsplit', '')
	first = kwargs.get('first', '')
	last = kwargs.get('last', '')


	if os.path.isdir(path):
		# parts = urllib.parse.urlsplit(self.path)
		# if not parts.path.endswith('/'):
		# 	# redirect browser - doing basically what apache does
		# 	self.send_response(HTTPStatus.MOVED_PERMANENTLY)
		# 	new_parts = (parts[0], parts[1], parts[2] + '/',
		# 					parts[3], parts[4])
		# 	new_url = urllib.parse.urlunsplit(new_parts)
		# 	self.send_header("Location", new_url)
		# 	self.send_header("Content-Length", "0")
		# 	self.end_headers()
		# 	return None
		for index in "index.html", "index.htm":
			index = os.path.join(path, index)
			if os.path.exists(index):
				path = index
				break
		else:
			# return self.list_directory(path)
			self.send_error(HTTPStatus.NOT_FOUND, "File not found")
			return None

	# check for trailing "/" which should return 404. See Issue17324
	# The test for this was added in test_httpserver.py
	# However, some OS platforms accept a trailingSlash as a filename
	# See discussion on python-dev and Issue34711 regarding
	# parseing and rejection of filenames with a trailing slash
	if path.endswith("/"):
		self.send_error(HTTPStatus.NOT_FOUND, "File not found")
		return None



	# else:

	return self.send_file(path, cache_control="no-cache")



























def AUTHORIZE_POST(req: SH, post:DPD, post_type=None):
	"""Check if the user is authorized to post

	reads upto line 5
	# args:
		req: the request handler (self of server handler class
		post: instance of DealPostData class
		post_type: post type to check
	"""

	# START POST DATA READING
	post.start()
	form = post.form


	post_verify = form.get_multi_field(verify_name="post-type", verify_msg=post_type, decode=T)


	##################################

	# HANDLE USER PERMISSION BY CHECKING UID

	##################################
	print(req.req_hash, "|=>> Post type:", post_verify[1])

	return post_verify[1] # return 1st field value


def Get_User_from_post(self: SH, post:DPD, pass_or_uid='password') -> Tuple[str, str]:
	"""
	Get username and password from post data
	READS UPTO LINE 13
	# args:
		self: the request handler (self of server handler class
		post: instance of DealPostData class
		pass_or_uid: verify using password or uid
		"""

	form = post.form

	_, username = form.get_multi_field('username', decode=T) # line 5-8
	username = username


	_, password = form.get_multi_field(pass_or_uid, decode=T) # line 9-12
	password = password

	return username, password

def resp_json(success, message='', **kwargs):
	"""
	returns json.dumps string based on success of an post action
	# args:
		success: bool or string refering wheather the task was successful or not
		message: any message or info the server wants to send to the front end
	"""
	out = {"status": success, "message": message}
	out.update(kwargs)
	return json.dumps(out)

def auth_uname_pass_data(uname, pw, max_pw=64):
	"""
	check if username and pass has weird data
	"""
	valid_uname_re = re.compile(r"[^a-zA-Z0-9_]")
	# print([uname, pw])
	if len(uname)==0 or valid_uname_re.search(uname) or len(pw)>max_pw:
		return False

	return True


def add_user_cookie(user_name, uid):
	# add cookie with 1 year expire
	cookie = SimpleCookie()
	def x(k, v):
		nonlocal cookie
		cookie[k] = v
		cookie[k]["expires"] = 365*86400
		cookie[k]["path"] = "/"

	x("uname", user_name)
	x("uid", uid)


	return cookie


@SH.on_req('POST', url='/login', hasQ='do_login')
def do_login(self: SH, *args, **kwargs):
	"""
	handle log in post request.
	1st validate post
	2nd get user from request username and *password*
	2.1 if username or password invalid, then Get_User_from_post(...) will send invalid request error and this will return None
	3rd sends username pass to user_handler and the handler will return if the action was successful or not and a message
	"""
	post = DPD(self)

	# check cookie


	AUTHORIZE_POST(self, post, 'login')

	username, password = Get_User_from_post(self, post)

	validity = auth_uname_pass_data(username, password)
	if validity !=True:
		return self.send_json(resp_json(validity))

	data = user_handler.server_login(username, password)
	if data["status"] == "success":
		cookie = add_user_cookie(data["user_name"], data["user_id"])

		self.send_response(200)
		self.send_header_string(cookie.output())

	return self.send_json(data)



@SH.on_req('POST', url='/signup', hasQ='do_signup')
def do_signup(self: SH, *args, **kwargs):
	"""
	signup user
	same as `do_login(...)`
	"""
	post = DPD(self)

	AUTHORIZE_POST(self, post, 'signup')

	username, password = Get_User_from_post(self, post)

	validity = auth_uname_pass_data(username, password)
	if validity !=True:
		return self.send_json(resp_json(validity))

	data = user_handler.server_signup(username, password)
	if data["status"] == "success":
		cookie = add_user_cookie(data["user_name"], data["user_id"])

		self.send_response(200)
		self.send_header_string(cookie.output())
	print(data)
	return self.send_json(data)


@SH.on_req('POST', hasQ='do_verify')
def do_verify(self: SH, *args, **kwargs):
	"""
	verify user
	same as `do_login`
	"""
	post = DPD(self)

	AUTHORIZE_POST(self, post, 'verify')

	username, uid = Get_User_from_post(self, post, 'uid')
	# uid (sha1) length is 40

	validity = auth_uname_pass_data(username, uid, 40)
	if validity !=True:
		return self.send_json(resp_json(validity))

	x = resp_json(user_handler.server_verify(username, uid))
	# print(x)
	return self.send_json(x)



@SH.on_req('POST', hasQ='bot_manager')
def bot_manager(self: SH, *args, **kwargs):
	"""
	handles user based varius bot queries like bot background, skin texture url etc.


	"""
	post = DPD(self)

	request  = AUTHORIZE_POST(self, post)

	username, uid = Get_User_from_post(self, post, 'uid')



	if request == 'get_skin_link':
		skin = user_handler.get_skin_link(username, uid)
		success = bool(skin)
		return self.send_json(resp_json(success, skin))

	elif request == 'room_bg':
		skin = user_handler.room_bg(username, uid)
		success = bool(skin)
		return self.send_json(resp_json(success, skin))

	else:
		return self.send_error(HTTPStatus.BAD_REQUEST, "Invalid request")


@SH.on_req('POST', url='/chat', hasQ='send_msg')
def chat(self: SH, *args, **kwargs):
	"""
	handles messaging to bot.

	gets message and sent time from post
	and send it to message handlerq
	"""
	post = DPD(self)

	AUTHORIZE_POST(self, post)
	form = post.form

	username, uid = Get_User_from_post(self, post, 'uid')



	_m, message = form.get_multi_field('message', decode=T)
	message = message.strip()

	_t, _time = form.get_multi_field('time', decode=T)
	_time = _time.strip()

	_tz, _time_offset = form.get_multi_field('tzOffset', decode=T)
	_time_offset = _time_offset.strip()

	_v, _voice = form.get_multi_field('voice', decode=T)
	_voice = json.loads(_voice) # bool


	if not _m or not _t:
		raise PostError("Invalid post data")


	out = {
		"status": "success",
		"message": '',
		"mid": 1, # message id
		"rid": 1, # reply id
		"delay": 0, # delay in seconds
	}
	# print("Message from %s: %s"%(username, msg))

	user = user_handler.collection(username, uid)
	if not user:
		out["status"] = "error"
		out["message"] = "User not found!\nPlease register first."
		out["script"] = ["""(async () => {
			await tools.sleep(3000);
			user.logout()
		})()"""]
		return self.send_json(out)

	user.browser_time_offset = int(_time_offset)/1000
	user.browser_time = int(_time)/1000

	# TODO: REMOVE THIS IN PRODUCTION
	importlib.reload(Chat_raw2)

	Chat_raw2.LOG_DEBUG = True

	reply = Chat_raw2.basic_output(message, user)

	if isinstance(reply, MessageObj):
		if "unknown" in reply.intents:
			_voice = False

		out_msg = reply
		out["message"] = reply.for_HTML()

	else:
		out_msg = MessageObj(user)
		out["message"] = reply


	if _voice:
		voice = reply.for_voice()

		voice_file = get_audio(voice, output_dir= appConfig.audio_file)
		# get file name from voice file
		voice_file = os.path.basename(voice_file)

		out["voice"] = f"/voice?id={voice_file}"

	out_msg.update(out)

	return self.send_json(out_msg.trimmed())


def main():
	if 0 and not check_internet():
		pass # now works
		xprint("/rh/No internet connection!\nPlease connect to the internet and try again.\n\n/=//hu/THIS APP IS HIGHLY DEPENDENT ON INTERNET CONNECTION!/=/")
		sys.exit(1)
	server_runner = pyrobox_runner(port= 45454,
		directory=appConfig.ftp_dir,
		bind="", # bind to all interfaces
		handler=SH)

	server_runner.run()

if __name__ == '__main__':
	main()

