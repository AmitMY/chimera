from datetime import datetime


class Time:
    @staticmethod
    def passed(start_time):
        return str(Time.now() - start_time)

    @staticmethod
    def now():
        return datetime.now()
