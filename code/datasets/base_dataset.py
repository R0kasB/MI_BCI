class BaseDataset:
    def __init__(self, subjects, sessions_per_subject, events, code, interval, doi, additional_info):
        self.subjects = subjects
        self.sessions_per_subject = sessions_per_subject
        self.events = events
        self.code = code
        self.interval = interval
        self.doi = doi
        self.additional_info = additional_info
        
    def __repr__(self):
        return (f"Dataset:\n"
                f"Subjects: {self.subjects}\n"
                f"Sessions per subject: {self.sessions_per_subject}\n"
                f"Events: {self.events}\n"
                f"Code: {self.code}\n"
                f"Interval: {self.interval}\n"
                f"DOI: {self.doi}\n"
                f"Additional info: {self.additional_info}")
'''
super().__init__(
    subjects = [],
    sessions_per_subject = [], 
    events = [],
    code = '',
    interval = [], 
    doi = '',
    additional_info = ''
)
'''