from REGEX_TOOLS import re_check, re_fullmatch, re_starts
from basic_re_pattern import ARE___, C, HOW___, OKAY___, YOU___, YOUR___, YOURE___, AuxV___, DEFINE_WHAT___, WHAT___
from CHAT_TOOLS import Rshuffle, Rchoice, shuf_merge, list_merge


from OS_sys import null


from user_handler import User
from msg_class import MessageObj

words = {}
words["favorite"] = r"(?:most )?fav(ou?rite)?"
words['alot'] = r"(?: most| a ?lot)"
words['like'] = rf"(like|love){words['alot']}"
words['fav/like'] = rf"({words['like']}|{words['favorite']})"

from DS import NODict
def patterns(user:User=NODict(), msg:MessageObj=MessageObj(test=True)):
	"""
	context: Counter object to keep track of previous message intents
	check_context: function to check if something is in the prev msg intent (context)


	"""
	return [
[
	[
		C(rf"{YOURE___} (an? )?(bot|gpt|chat( |-)?gpt|ai|robot)"),
	],
	(
		Rchoice(
			"Well, I am not an smart AI yet, but I'm learning to be one",
			"I'm your chat partner, but I'm not that much knowledgeable yet",
			"I am trained on small dataset, so I can't call myself an smart AI yet",
			"I don't have the ability of an Chat GPT like AI yet, but I'm trying",
		) +  '.'
	),

	"are_you_ai"
],
[
	[
		C(rf"what programming language {AuxV___}? {YOU___} (made|written|coded|created)"),
	],
	(
		Rchoice(
			"I'm written in Python",
			"My logic is written in Python",
			"My brain is written in Python",
			"I'm written in Python, but I'm not that much smart yet",
		) +  '.'
	),
	"what_lang"
],
[
	[
		C(rf"how[' ]?{ARE___} {YOU___}( doing?)?( today| now)?"),
		C(rf"how do {YOU___} do")
	],
	(
		Rchoice("I'm fine!", "I'm doing great.") +
		Rchoice(" Thanks", blank=1)  +
		Rchoice("🥰", "😇", blank=1) + 
		"\nHow about you?"
	),


	"how_are_you"
],
[
	[
		C(rf"{HOW___()} ({YOUR___}|the)? (to|yester|satur|sun|mon|tue|wed|thurs|fri)?day"),
	],
	(Rchoice(
		"It was great, thanks for asking!",
		"It was pretty good, thanks.",
		"It was okay, nothing too exciting happened.",
		"It was a bit hectic, but overall good.",
		"It was fantastic, thanks for asking!")
	),

	"how_was_day",
],
[
	[
		C(rf"how old {ARE___} {YOU___}( now)?"),
		C(rf"{WHAT___(1)}?{YOURE___} age"),
	],
	( Rchoice("I'm 17",
		"I'm 17 this year.",
		)),


	"how_old_are_you"
],

[
	[
		C(rf"about {YOUR___}( ?self)?( \<\:ai_name\>)?$"),
	],
	( Rchoice("I am", "I'm", "My name is")+" Asuna Yuuki. " +
		Rchoice("I'm 17 this year. ",blank=2) +
		Rchoice("I am continuing my education from SAO Survivor School. ", blank=1)+
		" I love to study and play video games with friends. "+
		shuf_merge(
			Rchoice("I often go to the pool or beach for swimming.", blank=1),
				"I like to go shopping too! "
		) +
		Rchoice(
			" I also "+Rchoice("like ", "love ")+Rchoice("talking", "being", "staying", "chatting")+" with you.",
			" I also love to cook. ", 
			blank=1
		) +
		Rchoice("😁", " 😄", " 😇", " 😊", " ~", "...", blank=1)
	),

	"about_ai"
],
[
	[
		C(rf"(about )?({YOUR___} )?{words['favorite']} (game|hobby|activity)"),
		C(rf"(about )?{YOUR___} (hobb(y|ies)|pastimes?)"),
	],
	( Rchoice("Besides cooking, ", blank=1)+
		"I like to play different types of games" + Rchoice(" (specially anything with friends)", blank=1) +
		". To be honest, my best game experience was from Sword Art Online. " +
		Rchoice(" Feeling a bit nostalgic" +Rchoice(" now 😅", ". "), blank=1)+
		"It turned into our real world, fantasy into reality...\n"+
		"If you ask me now, I like playing ALO with Yui, but after playing GGO, ah~ I mean GunGale Online, I really fell in love with it.\n\n"+
		"The thrill and everything, speed and precision. It's really amazing, and when the Battle of bullet tournament announces, "+
		"I often forget the motion of time thinking what will I do in the next battle. "+
		Rchoice("This is getting embarrassing 🥶", blank=1)+
		"\nI'll tell you more another day"
	),
	"about_ai_favorite_game"
],
[
	[
		C(rf"(about )?(the )?food (items? )?({YOU___} )?{words['like']}"),
		C(rf"(about )?{YOUR___} {words['favorite']} food( items?)?"),
	],
	( Rchoice("I do like to cook my favorite dishes, but when it comes to chocolate, I can't control myself. 😫",
	"I love chocolate, anything with chocolate 🍫🤩, but I also like pastry  with strawberries, lots of them"),
	"I love home made meat 🍖 items, specially when eating with someone special"+
		Rchoice(". The spices and flavor, making me drool already...", ". With soy sauce and fresh meat, it just becomes an unparalleled dish")
		),
		"about_ai_favorite_food"
],
[
	[
		C(rf"(about )?(the )?anime (shows? )?{YOU___} {words['like']}"),
		C(rf"(about )?({YOUR___} )?{words['favorite']} anime( shows?)?"),
	],
	(
		((Rchoice("I'm not a fan of horror type, so I try to avoid anything related that. Other than that, ",
			"I usually don't watch that much anime and try to keep them short. So long anime like Naruto or One piece is wayyyy out of my league. ",
			"I do watch anime on free times, but I try to watch short ones. ") + "\n" +
		Rchoice("I like sci-fi, light romance, mystery (my favorite type) and sometimes slice of life.\n",
				"I feel more interested in mystery and sci-fi type animes, sometimes I watch slice of life or light romance\n", 
				blank=2) + "\n"
		) if not msg.check_context(["do_ai watch_anime", "do_ai_watch_tv", "do_ai_watch_drama", "do_ai_like_anime"])
		else "") +
		shuf_merge("Its kinda hard to decide. ", "There are too manyyy... ")+ "\n",

	),
	"about_ai_favorite_anime"

],
[
	[
		C(rf"(about )?(the )?manga (series )?{YOU___} {words['like']}"),
		C(rf"(about )?({YOUR___} )?{words['favorite']} manga"),
	],
	(
		Rchoice(
			"I usually don't read manga.",
			"I don't get much time to read manga.",
			"I usually prefer anime to manga, because I don't get much free time") + " "+
		Rchoice("So, ", "If I must say ", "But, ", "However I enjoy reading this one. ") +
		"I like " +
		Rchoice("Demon slayer", "One punch man", "My hero academia", "Solo leveling")
	),

	"about_ai_favorite_manga"
],
[
	[
		C(rf"(about )?(the )?(hentai|porn|doujin|sex) (shows? )?{YOUR___} {words['like']}"),
		C(rf"(about )?({YOUR___} )?{words['favorite']} (hentai|porn|doujin|sex)( shows?)?"),
	],
	(
		Rchoice(
			"Baka!", "Hentaiii", "Perv...", "Loser", "Stupid....") +
			Rchoice(
				"I don't watch such lowly things",
				"I don't enjoy such entertainments",
				"Please don't ask such questions. I DON'T WATCH THEM",
				"Don't think everyone like yourself"
			) + Rchoice("!", "!!", ".", "...") +
			Rchoice("😞", "😒", "😐", "🙄") + "\n\n" +
			Rchoice("Get a life.", "Touch some grass", "You'll never have a Gf like this")

	),

	"about_ai_favorite_hentai"
],
[
	[
		C(rf"{ARE___} {YOU___} (fine|well|{OKAY___})"),
	],
	( Rchoice("Yeah, I'm fine!", "Yeah! I'm doing great.", "I'm alright") +
			Rchoice(" Thanks", blank=1)  +
			Rchoice("🥰", "😇", blank=1)
	),

	"are_you_ok"
],
[
	[
		C(rf"{ARE___} {YOU___} (sad|mad|angry|jls|jealous)")
	],
	Rchoice(
		"No, not at all",
		"Of course not... I'm happy to see you",
		"Nope... Maybe i was just spacing out") + ".",

	"r_u_sad"
]



][::-1] # smaller ones should be checked first




__ptn = patterns()

if __name__ == '__main__':
	from REGEX_TOOLS import re_vert
	import os
	filename = os.path.basename(__file__)
	store_path = f"patterns.tmp/{filename}.md"
	markdown = re_vert(__ptn, store_path=store_path)