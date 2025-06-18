import util


class RecognitionHandler:
    def __init__(self, db_dir, known_encodings=None, known_names=None, multi_encodings_dict=None):
        self.db_dir = db_dir
        self.known_encodings = known_encodings or []
        self.known_names = known_names or []
        self.multi_encodings_dict = multi_encodings_dict or {}

    def reload_known_faces(self):
        self.known_encodings, self.known_names, self.multi_encodings_dict = util.load_known_faces(self.db_dir)

    def recognize_face(self, frame, use_multi_encodings=False):
        # Calls util.recognize; returns (status, emp_id or name)
        return util.recognize(
            frame,
            self.db_dir,
            self.known_encodings,
            self.known_names,
            use_multi_encodings=use_multi_encodings
        )