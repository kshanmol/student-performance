import sys
import os

"""
1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
0 - GP
1 - MS

2 sex - student's sex (binary: "F" - female or "M" - male)
0 - F
1 - M

4 address - student's home address type (binary: "U" - urban or "R" - rural)
0 - U
1 - R

5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
0 - LE3
1 - GT3

6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
0 - T
1 - A

9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
(D1, D2, D3, D4)
0000 - teacher
0001 - health-care related
0010 - civil services
0100 - at home
1000 - other

10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
(D1, D2, D3, D4)
0000 - teacher
0001 - health-care related
0010 - civil services
0100 - at home
1000 - other

11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
(D1, D2, D3)
000 - close to home
001 - reputation
010 - course preference
100 - other

12 guardian - student's guardian (nominal: "mother", "father" or "other")
(D1, D2)
00 - mother
01 - father
10 - other

16 schoolsup - extra educational support (binary: yes or no)
0 - yes
1 - no

17 famsup - family educational support (binary: yes or no)
0 - yes
1 - no

18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
0 - yes
1 - no

19 activities - extra-curricular activities (binary: yes or no)
0 - yes
1 - no

20 nursery - attended nursery school (binary: yes or no)
0 - yes
1 - no

21 higher - wants to take higher education (binary: yes or no)
0 - yes
1 - no

22 internet - Internet access at home (binary: yes or no)
0 - yes
1 - no

23 romantic - with a romantic relationship (binary: yes or no)
0 - yes
1 - no
"""

def process(file_name_string, dummy_values):

	file_name = os.path.join(os.path.dirname(__file__), 'data/'+ file_name_string)
	output_file_name = os.path.join(os.path.dirname(__file__), 'data/'+ 'transformed-' + file_name_string)
	output_data = []

	with open(file_name, 'r') as f:
		headers = f.readline()
		for line in f:
			data = line.split(';')
			transformed_data = []
			for idx,item in enumerate(data):

				item = item.strip("\"")
				item = item.strip("\n")

				if(idx+1 not in dummy_values):
					transformed_data.append(int(item))
				else:
					values = dummy_values[idx+1][item]
					if isinstance(values, list):
						transformed_data.extend(values)
					else:
						transformed_data.append(values)
			result = ",".join(map(str, transformed_data)) + '\n'
			output_data.append(result)

	with open(output_file_name, 'w') as f:
		for item in output_data:
			f.write(item)


def main():

	# For each column number,
	# A map of column value to value of dummy variable(s)

	dummy_values = {}
	dummy_values[1] = {"GP":0, "MS": 1}
	dummy_values[2] = {"F":0, "M":1}
	dummy_values[4] = {"U":0, "R":1}
	dummy_values[5] = {"LE3":0, "GT3":1}
	dummy_values[6] = {"T":0, "A":1}
	dummy_values[9] = {"teacher":[0,0,0,0], "health":[0,0,0,1],
					"services":[0,0,1,0], "at_home":[0,1,0,0], "other":[1,0,0,0]}
	dummy_values[10] = {"teacher":[0,0,0,0], "health":[0,0,0,1],
					"services":[0,0,1,0], "at_home":[0,1,0,0], "other":[1,0,0,0]}
	dummy_values[11] = {"home":[0,0,0], "reputation":[0,0,1],
						"course": [0,1,0], "other":[1,0,0]}
	dummy_values[12] = {"mother":[0,0], "father":[0,1], "other":[1,0]}
	dummy_values[16] = {"yes":0, "no":1}
	dummy_values[17] = {"yes":0, "no":1}
	dummy_values[18] = {"yes":0, "no":1}
	dummy_values[19] = {"yes":0, "no":1}
	dummy_values[20] = {"yes":0, "no":1}
	dummy_values[21] = {"yes":0, "no":1}
	dummy_values[22] = {"yes":0, "no":1}
	dummy_values[23] = {"yes":0, "no":1}

	process('student-mat.csv', dummy_values)
	process('student-por.csv', dummy_values)


if __name__ == '__main__':
	main()