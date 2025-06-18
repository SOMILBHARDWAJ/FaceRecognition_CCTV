import time

# Dictionary to store timer values for each user
userTimers = {}

# Called when a user is recognized or not
def update_attendance(user_id, is_present):
    if user_id not in userTimers:
        userTimers[user_id] = {
            'presentCounter': 0,
            'absentCounter': 0,
            'absentTimeCounter': 0,
            'lastUpdateTime': time.time()
        }

    currentTime = time.time()
    elapsedTime = currentTime - userTimers[user_id]['lastUpdateTime']

    # Only update every 5 seconds
    if elapsedTime >= 5:
        timers = userTimers[user_id]

        if is_present:
            # If user was absent for < 30s, add that back to present time
            if 0 < timers['absentCounter'] < 30:
                timers['presentCounter'] += timers['absentCounter']
            # Add the current 5 seconds of presence
            timers['presentCounter'] += 5
            # Reset absence counter
            timers['absentCounter'] = 0
        else:
            # Accumulate absence
            timers['absentCounter'] += 5
            # If absence reaches 30s, count it as missed time
            if timers['absentCounter'] >= 30:
                timers['absentTimeCounter'] += 30
                timers['absentCounter'] = 0  # reset

        timers['lastUpdateTime'] = currentTime


# Get user timer data for UI display
def get_user_timer_data(user_id):
    if user_id in userTimers:
        return userTimers[user_id]
    return {
        'presentCounter': 0,
        'absentCounter': 0,
        'absentTimeCounter': 0
    }
