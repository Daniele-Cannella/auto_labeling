import platform
import os.path
from datetime import datetime
from icecream import ic


f_log = "../log/trace.log"

class Log:
    
    def __init__(self, nome_f: str):
        self.nome_f = nome_f

    def log(self, term):
        '''
        Funzione che aggiorna il file trace quando il programma
        viene eseguito e terminato.
        '''
        try:
            with open(f_log, "a") as flog:
                current_time = datetime.now().strftime("%H:%M:%S del giorno %m/%d/%Y")
                uname = platform.uname()

                if term is False:
                    log_message = f"Programma iniziato alle ore {current_time}"
                    log_entry = (
                    "----------------------------------------------\n"
                    f"{current_time}, {uname.node}, {uname.system}, {self.nome_f}, {log_message}\n"
                )
                else:
                    log_message = f"Programma terminato alle ore {current_time}"
                    log_entry = (
                    f"{current_time}, {uname.node}, {uname.system}, {self.nome_f}, {log_message}\n"
                )

                flog.write(log_entry)
        except Exception as ex:
            os.makedirs(os.path.dirname(f_log), exist_ok=True)
            self.write_error(ex)

    def write_error(self, error: str):
        current_time = datetime.now().strftime("%H:%M:%S del giorno %m/%d/%Y")
        with open(f_log, 'a') as file:
            file.write(f"File: {self.nome_f}, Error: {error}, Time: {current_time}\n")
            
    def write_msg(self, msg: str):
        current_time = datetime.now().strftime("%H:%M:%S del giorno %m/%d/%Y")
        with open(f_log, 'a') as file:
            file.write(f"File: {self.nome_f}, Message: {msg}, Time: {current_time}\n")


if __name__ == "__main__":
    nome_f = os.path.basename(__file__)
    logs = Log(nome_f)
    logs.log(False)

    # Your program logic here

    logs.log(True)
