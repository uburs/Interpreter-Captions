import subprocess

def analyze_intonation(audio_path, output_csv="intonation_features.csv"):
	smile_config = "IS09_emotion.conf"
	command = f'SMILExtract -C {smile_config} -I {audio_path} -O {output_csv}'
	subprocess.run(command, shell=True)
	print(f"Intonation features saved to {output_csv}")
	
analyze_intonation("speech.wav")
