import re
from re import compile
import traceback


from REGEX_TOOLS import re_starts, re_check, re_fullmatch, re_search, eos, eol, C
from DS import GETdict

from CHAT_TOOLS import merge

IM___ = r"i(?:'?| a)m"
ARE___ = r"(?:are|re?)"

YOU___ = r"((?:yo)?u|y[ao])"
YOUR___ = rf"({YOU___}r)"
YOURE___ = rf"({YOU___}[' ]?{ARE___}?)"


AuxV___ = (
	"(?:"
		"'?(?:"
			"m|s|re?|ll|ve"
		")"
		""
		"|"
		" (?:"
			rf"is|{ARE___}|was|were|am|will"
		")"
	")"
	"(?:"
		" ha(?:ve|d|s)"
	")?"
	"(?:"
		" be(?:en|ing)?"
	")?"
)

AuxV1___ = (
	"(?:"
		"'?(?:"
			"m|s|re?"
		")"
		""
		"|"
		" (?:"
			rf"is|{ARE___}|am"
		")"
	")"
)

AuxV2___ = (
	"(?:"
		"'?(?:"
			"re?"
		")"
		""
		"|"
		" (?:"
			rf"w(?:as|ere)"
		")"
	")"
)

AuxV3___ = (
	"(?:"
		"'?(?:"
			"ll"
		")"
		""
		"|"
		" (?:"
			rf"will|shall"
		")"
	")"
)

An___ = r"(?:an?)"

CHANGE___ = r"(change|swap|switch)"
PLEASE___ = r"((?:p[lw](?:ease|z|s)e?)|kindly)"
WILL_U___ = rf"(will {YOU___}(?: eve(?:n|r)){{0,2}})"
CAN_U___ = rf"(can {YOU___}(?: eve(?:n|r)){{0,2}})"
CANT___ = r"(can[' ](?:no)t)"
DO_U___ = rf"((?:do|did) {YOU___}(?: eve(?:n|r)){{0,2}})"

REQUESTING___ = "(" + '|'.join([WILL_U___, CAN_U___]) + ")"
ASKING___ = "(" + '|'.join([CAN_U___, DO_U___, WILL_U___]) + ")"

What___ = r"w(?:h|g)?[au]t"
Who___ = r"w(?:h|g)?o"
When___ = r"w(?:h|g)?en"
How___ = r"how"
Where___ = r"w(?:h|g)?ere"


DEFINE_WHAT___ = (
	"("
		"(?:"
			"about" # about your name
			"|"
			"define" # define this or that
			"|"
			f"{What___}( {AuxV___})?" # what is ....
		")"
		"(?: the)?" #define the ..., what is the ..
	")"
)

def AuxV(tense=0):
	aux = AuxV1___ if tense==1 else AuxV2___ if tense==2 else AuxV3___ if tense==3 else AuxV___

	return aux

def WHAT___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		 # what is ....
		f"( {aux})?"
		"(?: the)?" #define the ..., what is the ..
	")"
)


def WHO___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		f"{Who___}( {aux})?"
		"(?: the)?"
	")"
)


def WHEN___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		f"{When___}( {aux})?"
		"(?: the)?"
	")"
)


def HOW___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		f"{How___}( {aux})?"
		"(?: the)?"
	")"
)


def WHERE___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		f"{Where___}( {aux})?"
		"(?: the)?"
	")"
)


# WHO_WHAT___ = rf"((who|what){AuxV___})(?: the)?" # who is, what is
# WHEN_WHAT___ = rf"(when|what){AuxV___}(?: the)?" # when is, what is

def WHO_WHAT___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		f"{Who___}|{What___}"
		f"{aux}?"
		"(?: the)?"
	")"
)

def WHEN_WHAT___(tense=0):
	aux = AuxV(tense)
	return (
	"("
		f"{When___}|{What___}"
		f"{aux}?"
		"(?: the)?"
	")"
)

DRESS___ = r"(dress|cloth|skin|costume|wear)(?:e?s)?"
ROOM___ = r"(room|place|location|background|bg)"

YES___ = r"(y(?:e|a)(?:ah|s|p))"
OKAY___ = r"(ok(?:(?:a|e)y|h|eh|ie?)?|alright)"

to_bot_suffix = C(
	"("
		f"{PLEASE___}? ?"
		"("
			"<:ai_name>|"
			"girl|"
			"dear|"
			"babe|"
			"honey|"
			"sweet ?heart|"
			"darling|"
			"ma.?am"
		")?"
	")"
	"$" # make sure line ends with it
)


def remove_suffix(string):
	return to_bot_suffix.sub("", string)




"""yes = "y", "yes", "yeah", "sure", "ok", "lets go", "let's go", "start", "yep", "yeap"
yes2 = yes1 = yes
yes+=tuple('well ' + j for j in yes2)
yes+=tuple('actually ' + i for i in yes1)"""
yes = (
	'y', 'yes', 'yeah', 'sure', 'ok', 'lets go', "let's go",
	'start', 'yep', 'yeap', 'well y', 'well yes', 'well yeah',
	'well sure', 'well ok', 'well lets go', "well let's go",
	'well start', 'well yep', 'well yeap', 'actually y',
	'actually yes', 'actually yeah', 'actually sure',
	'actually ok', 'actually lets go', "actually let's go",
	'actually start', 'actually yep', 'actually yeap'
)


"""no = ("n", "no", "na", "nah", "nope", "stop", "quit", "exit", 'not really', 'no', 'not at all', 'never')
no2 = no1 = no
no+=tuple('well ' + j for j in no2)
no+=tuple('actually ' + i for i in no1)"""

class atdict(dict):
	__getattr__= dict.__getitem__
	__setattr__= dict.__setitem__
	__delattr__= dict.__delitem__


ot = atdict()

ip = atdict()

it = atdict()


ot.yes = ("Yeah!", "Sure...", "Sure!!", "Okkay~", "Okie~", "Okay!")
ot.no = ("No", "Sorry but nope")
ot.tell_time = ('The time is ', "It's ")
ot.tell_date = ("Today is", "It's")

happy_emj = ("(◕‿◕)💞", "😄",
				"😇", "😊", "~", "...", "", "")
sad_emj = ("😿", "😢", "😭",
			  "😞", "😔", "~", "...", "", "")
ot.happy_emj = happy_emj
ot.sad_emj = sad_emj
ot.my_name_is = ["My name is ", "I am ",
				 "Its ", "Call me ", "You can call me "]
ot.call_me = ["You can call me ", "Call me ", "Its "]
ot.about_self = ('I am your virtual partner. My name is <:ai_name> and I was made by <a href="https://github.com/RaSan147/VoiceAI-Asuna" target="_blank">RaSan147</a>',
				 'I am an AI. My name is <:ai_name> & I am your voice assistant.', 'My name is <:ai_name>. I am an AI voice assistant.')

ot.on_whats_up = (
	"Just the usual.",
	"Nothing much.",
	"Nothing much, just chilling.",
	"Nothing much, just hanging around.",
	"All good here!",
	"I'm doing well.",
	"Nothing much, just doing my thing.",
)


ot.no_internet = (
	"Sorry, server is offline right now.",
	"Its kinda embarrassing to say, the server is facing internet outage."
	"Something is wrong with out network. We're working on it."
)

ot.internet_ok = (
	"Internet connection is available",
	"You're online",
	"The network is stable",
	"Your network is online"
)

ip.logout = [
	C(r"(log|sign)o? ?(out|off)"),
]


ip.yeses = [
	C(r'(well )?(actually )?y(e|a)(ah|s|p)( of ?course)?( sure)?'),
	# well actually yes/yep/yeah of~course
	'sure',
	'of( |-)?course',
	C(r"ok+(ay|h|eh)?"),  # okkay/okeh
	"go (on|ahead)",
	C("^y$")
]

# print(ip["yeses"][3].match("okkeh"))

no = ('n', 'no', 'na', 'nah', 'nope', 'stop', 'quit', 'exit', 'not really', 'no', 'not at all', 'never', 'well n', 'well no', 'well na', 'well nah', 'well nope', 'well stop', 'well quit', 'well exit', 'well not really', 'well no',
	  'well not at all', 'well never', 'actually n', 'actually no', 'actually na', 'actually nah', 'actually nope', 'actually stop', 'actually quit', 'actually exit', 'actually not really', 'actually no', 'actually not at all', 'actually never')

cond = yes + no
ip.no = [
	# well actually no/nope/not/nah // not at all! never!!!
	C(r"(well )?(actually )?n(o+(pe)?t?|ah?)( at all)?( never)?"),
	C(rf"{PLEASE___}?stop"),
	"never",
]


# print(ip["no"][0].match("well nope"))
# print(fullmatch(ip["no"], "well nope"))

li_QyuiName = "can i change your name", 'i want to change your name'
li_QyuiNamePre = "can i call you ", 'may i call you'
# li_hello = "hello <:ai_name>", "helo <:ai_name>", 'hello', 'helo'
# li_hi = "hi <:ai_name>", "hey <:ai_name>", 'hi', 'hey', "hiii"

li_redo = 'redo my last command', 'retry my last command', 'redo last command', 'redo last command', 'redo'

ip.r_u_ok = [
	C(
		f"{ARE___} {YOU___}"
		"(?: feeling)?"
		" (?P<OK___>" # can be used in reply
			"fine"
			"|ok(?:[ae]y)?"
			"|well"
			"|alright"
		")"
	),
]

ip.thanks = [
	C(
		"("
			"thanks?"
			"(?: "
				"a (?:lot|bunch)"
				"|"
				"{YOU___}(?:"
						"(?: so| very)* much"
					")?" # so/very much (optional)
			")?"
		")"
	),
]

ip.r_u = [
	C(rf"{ARE___} {YOU___}"),
]
ip.who_are_you = [
	C(rf"who[' ]?{ARE___} {YOU___}"),  # who are u
]


ip.whats_ = [
	# C(r"((can ((yo)?u|y(a|o)) )?(please )?((tell|speak|say)( me)? )|((do|did) )?((yo)?u|y(a|o)) know )?(what ?(s|re|is|are|was|were)? )(the )?(?P<query>.*)"),
	C(rf"{DEFINE_WHAT___} (?P<query>.*)"),
]

ip.whos_ = [
	# C(r"((can ((yo)?u|y(a|o)) )?(please )?((tell|speak|say)( me)? )|((do|did) )?((yo)?u|y(a|o)) know )?(what ?(s|re|is|are|was|were)? )(the )?(?P<query>.*)"),
	C(rf"{WHO___()}(?P<query>.*)"),
]

ip.whens_ = [
	C(rf"{WHEN___()}(?P<query>.*)"),
]

ip.hows_ = [
	C(rf"{HOW___()}(?P<query>.*)"),
]

ip.wheres_ = [
	C(rf"{WHERE___()}(?P<query>.*)"),
]

ip.whats_your_name = [
	# C(r"((can ((yo)?u|y(a|o)) )?(please )?((tell|speak|say)( me)? )|((do|did) )?((yo)?u|y(a|o)) know )?(what(s|re| (is|are|was|were))? )?((yo)?u|y(a|o))(r|re)? name"),
	C(rf"({WHAT___()})?{YOUR___} name"),
	# C(r"((((can|will) ((yo)?u|y(a|o)) )?(please )?)?(tell|speak|say) (me )?)?what should i call ((yo)?u|y(a|o))( by)?")

]



ip.what_to_call_you = [
	C(rf"{WHAT___()} should i call {YOU___}( by)?"),
]

ip.what_time = [
	# C(r"((can ((yo)?u|y(a|o)) )?(please )?((tell|speak|say)( me)? )|((do|did) )?((yo)?u|y(a|o))( even)? know )?(what(s|re| (is|are|was|were))? )?(the )?(current )?time( is| it)*( now)?( please)?"),
	C(
		"("
			"(?:"
				f"{WHAT___()} ?"
			")?"
			"(?:the )?"
			"(?:current )?"
			"time"
			f"{eos}" # end of sent. or nexr word
			"(?:is|it)*"
			"(?: now)?"
			f"(?: {PLEASE___})?"
		')'
	),
	'clock',
]

ip.what_date = [
	# C(r"((can ((yo)?u|y(a|o)) )?(please )?((tell|speak|say)( me)? )|((do|did) )?((yo)?u|y(a|o))( even)? know )?(what(s|re| (is|are|was|were))? )?(the )?(current )?time( is| it)*( now)?( please)?"),
	C(
		'('
			f"(?:{WHAT___()} )?"
			"(?:current )?"
			"date"
			f'{eos}'
			"(?: is| it)*"
			"(?: now)?"
			f"(?: {PLEASE___}?)"
		')'
	),
	C('cal[ae]nd[ae]r'),
]

li_how_old_r_u = 'old are you', 'your age'
li_where_r_u = 'you',
li_where_r_u_frm = 'you from',

li_WmyName = 'my name',


"whats my name"
it.my_name = ['my name', 'my nickname']


"what/who am i to you?"
ip.my_self = [
	C(rf"me( to {YOU___})?"),
	C(rf"my ?self( to {YOU___})?"),
	C(rf"i( to {YOU___})?")
]


"what are you?"
ip.you_self = [
	C(rf"{YOURE___}( ?self)?( really)?"),
]

ip.your_bday = [
	C(rf"{YOUR___} (birth|b) ?day")
]


"whats the latest news" "tell me the latest news"
ip.latest_news = [
	C(r'(latest|any|global|world)? ?(news|special|) ?(news|headlines?|events?|updates?)'),
]

ip.tell_latest_news = ip.latest_news + [
	"anything interesting happening",
	C(rf"(do {YOU___})? ?(got|have) (some|any)(thing)? (news|headlines?|interesting)(today)?")
]


'((tell|speak|read )(out)?)?(the )?(latest )?news'

"change Anime dress"

ip.change_cloth = [
	'change',
	C(rf"{CHANGE___} (((yo)?u|y(a|o))('| )?(re?)? )?{DRESS___}"),
	C(rf"wear (a )?(((yo)?u|y(a|o))('| )?(re?)? )?(new )?{DRESS___}"),
]

"change Anime room"

ip.change_room = [
	C(rf"({CHANGE___}|move) (to )?(a )?(((yo)?u|y(a|o))('| )?(re?)? )?(new )?{ROOM___}"),
]


li_AmyName = 'Your name is ',

# print(li_whats)


###############################################################
# _li_time =('time', 'the time', 'current time')

# li_time = tuple(merge(what, time) for time in _li_time for what in li_what_is)
# # print(li_time)
# li_time1= tuple(merge("tell me",what, i) for i in _li_time for what in li_what_is2)
# li_time1+= tuple(merge("tell me", i) for i in _li_time)
# li_time1+= tuple(merge("tell", i) for i in _li_time)
# li_time1+= tuple(merge("say", i) for i in _li_time)
# li_time1+= tuple(merge("speak", i) for i in _li_time)

# li_time2= tuple(merge("can",y, i) for i in li_time1 for y in li_you)
# li_time2+= tuple(merge("can",y,"please", i) for i in li_time1 for y in li_you)
# li_time2+= tuple(merge("please", i) for i in li_time1)


# li_time3= tuple(merge(i, 'please') for i in li_time)
# li_time3+= tuple(merge(i, 'please') for i in li_time1)

# li_time+= li_time1
# li_time+= li_time2
# li_time+= li_time3

# li_time = tuple(set(li_time))
# print(tuple(set(li_time)))
#############################################################


ip.goto = [
	C(r"(open|go ?to)( the)? (?P<query>.*?)( website| site| page)?$")
]

ip.search = [
	C(r"(search|find|chack) (the )?(?P<query>.*?)")
]


li_tell_time2 = ('The time is ', "It's ")
li_goto = ('open', 'go to', 'goto')
li_play = ('play', 'lets play', 'hit', 'tune', 'sing')
li_reload = ('re', 'reload', '11')
li_fucku = ('fuck you', 'fuck u', 'fuck ya')

ip.fuck_you = [
	C(r"(i('| | wi)ll )?(fuckh?|rape|torture|kill) ((yo)?u|y(a|o))(('| )?(re?)? ((mo(m|ther|mmy))|sis(ter)?))?"),
]
# this is terrible, i wish no one use this ever
ot.fuck_you = (
	"I don't like you.", 'How rude!', "You're mean!", "You're rude",
	"Please refrain from using such terms", "You're horrible", "I don't want to talk to you", "You're disgusting")


ip.love_you = [
	C(rf'(i )?(really )?(love|wuv) {YOU___}( so much| a lot)?'),
]

ip.hate_you = [
	C(rf"(i )?(really )?(hate|don('| )t like) {YOU___}"),
]

ip.whats_up = [
	C(r"wh?(u|a)t?( |')?s+ up+"),
	C(r'^sup' + eos),
]


ip.check_net = [
	C("check (wifi )?(inter)?net(work)?"),
	C("check ((wifi )?(inter)?net(work)? )?connection")
]


li_relove = 'love you too', 'love you so much', 'I love you too'
li_voice0 = ['silent', 'silence', 'shut up',
			 'turn off volume', 'stop speaking']
li_can_do = li_goto + li_play
works = ["talk", "calculate"]

mc_pause = ['pause', 'pause it', 'pause the song', 'pause the music']
mc_resume = ['resume', 'resume it', 'resume the song', 'resume the music',
			 'continue', 'continue the song', 'continue the music']
mc_stop = ['stop', 'stop it', 'stop the song', 'stop the music']
mc_replay = ['replay', 'replay the song', 'replay the music',
			 'restart', 'restart the song', 'restart the music']
mc_vol_down = ['volume down', 'lower the volume', 'lower volume', 'vol down']
mc_vol_up = ['volume '+i for i in ('up', 'higher')
			 ] + [i+' the volume' for i in ('raise', 'increase', 'higher')
				  ] + [i+' volume' for i in ('raise', 'increase', 'higher')]

li_window_manage = ("forcemin",
					"hide",
					"maximize",
					"minimize",
					"restore",
					"show")


condERR = "Sorry,  I can't understand what you are saying. Just type yes or no.   "
nameGlad = "Ok. Glad to hear that you like my name."

ip.set_timer_pattern = "set ?a? timer of (.*)"

# db = generate_list('li_')

ip.bye = [
	"exit", "close",
	C(r"(shut|turn) ?(down|off)"),
	"quit",
	C(r"(good )?(bye+ ?)+"),
	C(r"esc(ape)?"),
	C(r"ta( |-)?ta"),
	C(r"see ((yo)?u|y(a|o))"),
]

ot.bye = "Bye", "See ya", "Take care", "See you later", "Good bye", "Good bye!", "Good bye..."

ip.take_care = [
	C(r"take( |-)?care"),
	C(r"sweet( |-)?dreams?"),
	"have a nice day",
]

ip.help = [
	C(r"[/\\]?(show(-| )?)?(help|commands?|menu)"),
]

ip.slur = [
	"fuck", "shit", "damn",
	C(r"bull.?shit"),
	"wtf", "tf",
	"suck",
]

ot.slur = (
	"What?! something wrong?",
	"Did I say something weird..."
	"Did I do something wrong...",
	#"I'm not that smart, if you have any problem please say, I'll try to fix myself.",
	"Something bad happened?"
)




# start_parrot = "parrot", "repeat after me", "repeat what i say", "mimic", "mimic me", "parrot mode", "parrot on", "turn parrot on", "start parrot", "start mimic", "start mimicing", "start mimicing me", "start mimicing me", "reply what i say", 'reply what i send', "copy me"

ip.start_parrot = [
	C(r"(start )?parrot( ?mode)?( on)?"),
	C(r"mimic( me)?"),
	C(r"start mimicking( me)?"),
	C(r"(re(ply|peat)|say) what i (say|type|send|write)"),
	C(r"repeat after me")
]
stop_parrot = "stop", "stop it", "stop mimicing", "stop mimic", "stop parrot", "off", "turn off", "turn parrot off", "cancel", "cancel mimic", "cancel parrot"

ip.stop_parrot = [
	C(r"(stop|cancel)( (it|mimicking|repeating|parrot))?"),
	C(r"(turn )?((parrot|it) )?of+"),
]

ot.created_by = ('I was %s by Ratul Hasan.',
		'Ratul Hasan %s me.',
		"I was %s by Rasan147 (Ratul Hasan)")


links_dict = {
	"url_google": ('https://www.google.com', 'google', 'gogle', 'gooogle'),
	"url_fb": ('https://www.facebook.com', 'facebook', 'facebok', 'fb'),
	"url_yahoo": ['https://www.yahoo.com', 'yahoo', 'yaho'],
	"url_youtube": ['https://www.youtube.com', 'youtube', 'tubemate', 'utube'],
	"url_wiki": ['https://www.wikipedia.com', 'wikipedia', 'wikipidia', 'wikipidea', 'wikipedea'],
	'url_reddit': ['https://www.reddit.com', 'reddit', 'redit'],
	'url_bing': ['https://www.bing.com', 'bing', 'microsoft search'],
	'url_insta': ['https://www.instagram.com', 'instagram', 'insta'],
	'url_apple': ['http://apple.com/', 'apple website', 'apple.com'],
	'url_microsoft': ['http://microsoft.com/', 'microsoft website', 'microsoft.com', 'microsoft site', 'microsoft page'],
	'url_pornhub': ['https://www.pornhub.com/', 'pornhub website', 'pornhub'],

	'goog_supp': ['http://support.google.com/', 'support', 'supports'],
	'goog_docs': ['http://docs.google.com/', 'doc', 'docs'],
}
# if 'insta' in links:print(links)
# sleep(10)
links = tuple(i for k, v in links_dict.items() for i in v)
links_li = tuple(v for k, v in links_dict.items())
# googles = generate_list('goog_')
# googles_li = gen_list('goog_')

# print(*links, sep='\n')






def preprocess(in_dat):
	""" replace . , " ' ? ! with space """
	# in_dat = in_dat.replace("'", " ")
	#in_dat = re.sub(r'[\?\!\,]| {2,}', ' ', in_dat)
	#in_dat = in_dat.replace("?", " ")
#	in_dat = in_dat.replace("!", " ")
#	in_dat = in_dat.replace(",", " ")
	in_dat = in_dat.strip()
#	in_dat = re.sub(r' {2,}', ' ', in_dat)
	# in_dat = in_dat.replace(" us ", " me")
	# in_dat = in_dat.replace(" him", " me")
	# in_dat = in_dat.replace(" her", " me")
	# in_dat = in_dat.replace(" them", " me")

	return in_dat


def pre_rem_bot_call(ui):
	"""
		* remove *hey* whats ....
		* remove *hey Asuna* whats ....

	"""
	nick = "<:ai_name>"
	ui = re.sub(
		rf'^(hey|miss|dear|yo)? ?(girl|babe|{nick})? ', '', ui, flags=re.IGNORECASE)

	ui = re.sub(rf'^{REQUESTING___} ', '', ui, flags=re.IGNORECASE)

	return ui


def post_rem_can_you(ui):
	"""
		0. remove `*can you* ....`
		1. remove `*will you* ....`
		2. remove `*do you know* ....`

		3. replace `*tell me* ....` with `....`
		4. remove `*tell me regarding* ....` with `*about* ....`
	"""
	ui = re.sub(rf'^((can|will|do|did) {YOU___})?( please| plz)?( even)? ?(know|tell|remember|speak|say)?( to)?( me)? (?P<msg>.+)',
				r'\g<msg>', ui, flags=re.IGNORECASE)
	ui = re.sub(r'^(of|regarding) ', 'about ', ui, flags=re.IGNORECASE)

	return ui
