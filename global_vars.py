# sadrži funkcije potrebne za rad s globalnom varijablom current_student

def init():
    global current_student
    current_student = 0


def increase():
    global current_student
    current_student += 1


def get_current_student():
    global current_student
    return current_student