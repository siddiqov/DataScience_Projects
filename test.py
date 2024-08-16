def calculate_final_speed(initial_speed, inclinations):
    for a in inclinations:
        
        if initial_speed<0:
            return "Life Lost"
        else:
            print("i am in else")
            if a==0:
                initial_speed=initial_speed+0
            elif a==30:
                initial_speed=initial_speed-30
            elif a==-45:
                initial_speed=initial_speed+45
        
    return(initial_speed)
        
if __name__=="__main__":
    print(calculate_final_speed(60, [0, 30, 0, -45, 0]))