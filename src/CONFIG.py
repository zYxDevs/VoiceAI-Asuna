class Config:
	def __init__(self):
		self.disable_lib_check = False
		self.data_dir = "./Alice_data/Users/" # DEFAULT DIRECTORY TO LAUNCH SERVER
		self.log_location = "./"  # fallback log_location = "./"
		self.temp_file = "Alice_data/temp/"

		self.cached_webpages_dir = self.temp_file + "cached_webpages/"


appConfig = Config()