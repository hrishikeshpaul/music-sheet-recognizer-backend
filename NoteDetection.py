# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:42:58 2020

@author: Sumanyu Garg (sumgarg)
"""
# In[1]:

import numpy as np

# In[1]:
def get_clef_positions(starting_positions):
    '''Ths function classifies if the staff is treble cleff or bass clef'''
    treble_clef, bass_clef = [], []
    for i in range(len(starting_positions)):
        if(i%2==0):
            treble_clef.append(starting_positions[i])
        else:
            bass_clef.append(starting_positions[i])
            
    return treble_clef, bass_clef
            

def belongs_which_clef(treble_clef, bass_clef, spacing_parameter,starting_positions, detected):
    ''' This function returns which notes belong to which clef
    0 denotes treble clef 
    1 denotes bass clef'''
    white = 255
    
    clef_dict = dict()
    for i in range(detected.shape[0]):
        for j in range(detected.shape[1]):
            if(detected[i,j]==white):
                #print(i,j)
                for clef in treble_clef:
                    if(clef - 3*spacing_parameter < i and i< clef + 8*spacing_parameter):
                        clef_dict[(i,j)] = 0
                        
                for clef in bass_clef:
                    if (i,j) not in clef_dict.keys():
                        clef_dict[(i,j)] = 1
    
    return clef_dict


def check_condition(i, position_dict_1, position_dict_2):
    '''This is a helper function to check if a note lies in a given range or not'''
    for x in range(len(position_dict_1)):
        condition = position_dict_1[x] < i and i < position_dict_2[x]
        if (condition==True):
            return True
        
    return False


def which_note_is_it(i, clef, max_error, note, first_encounter, spacing_parameter):
    '''This function specifies if a note is particular or not'''
    candidate_position_1 = [int(x + first_encounter - max_error) for x in clef]
    candidate_position_2 = [int(x + first_encounter + max_error) for x in clef]
    #print(candidate_position_1, candidate_position_2)
    condition_1 = check_condition(i, candidate_position_1, candidate_position_2)
    
    candidate_position_1 = [int(x + first_encounter + spacing_parameter*3.5 - max_error) for x in clef]
    candidate_position_2 = [int(x + first_encounter + spacing_parameter*3.5 + max_error) for x in clef]
    condition_2 = check_condition(i, candidate_position_1, candidate_position_2)
    
    candidate_position_1 = [int(x + first_encounter - spacing_parameter*3.5 - max_error) for x in clef]
    candidate_position_2 = [int(x + first_encounter - spacing_parameter*3.5 + max_error) for x in clef]
    condition_3 = check_condition(i, candidate_position_1, candidate_position_2)
    
    if(condition_1 or condition_2 or condition_3):
        return True
    
    return False
            

def create_notes_dict(clef_dict, treble_clef, bass_clef, spacing_parameter):
    ''' This function specifies which note each note is'''
    max_error = spacing_parameter/2
    notes = []

    for index,clef in clef_dict.items():
        if(clef==0):
            i = index[0]
            j = index[1]
            
            # Detecting which notes are F
            if(which_note_is_it(i, treble_clef, max_error, 'F', 0, spacing_parameter) == True):
                notes.append((i,j,'F'))
            
            # Detecting which notes are D
            elif(which_note_is_it(i, treble_clef, max_error, 'D', spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'D'))
            
            # Detecting which notes are B
            elif(which_note_is_it(i, treble_clef, max_error, 'B', 2*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'B'))
              
            # Detecting which notes are G
            elif(which_note_is_it(i, treble_clef, max_error, 'G', 3*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'G'))
            
            # Detecting which notes are E
            elif(which_note_is_it(i, treble_clef, max_error, 'E', 4*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'E'))
                
            # Detecting which notes are A
            elif(which_note_is_it(i, treble_clef, max_error, 'A', 2.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'A'))
                
            #Detecting which notes are C
            elif(which_note_is_it(i, treble_clef, max_error, 'C', 1.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'C'))
                
            # Whatever is left
            else:
                l = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                notes.append((i,j,np.random.choice(l)))
                
        if(clef==1):
            i = index[0]
            j = index[1]
            
            # Detecting which notes are A
            if(which_note_is_it(i, bass_clef, max_error, 'A', 0, spacing_parameter) == True):
                notes.append((i,j,'A'))
                
            # Detecting which notes are F
            elif(which_note_is_it(i, bass_clef, max_error, 'F', spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'F'))
                
            # Detecting which notes are D
            elif(which_note_is_it(i, bass_clef, max_error, 'D', 2*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'D'))
                
            # Detecting which notes are B
            elif(which_note_is_it(i, bass_clef, max_error, 'B', 3*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'B'))
                
            # Detecting which notes are G
            elif(which_note_is_it(i, bass_clef, max_error, 'G', 4*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'G'))
                
            # Detecting which notes are C
            elif(which_note_is_it(i, bass_clef, max_error, 'C', 2.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'C'))
                
            # Detecting which notes are E
            elif(which_note_is_it(i, bass_clef, max_error, 'E', 1.5*spacing_parameter, spacing_parameter) == True):
                notes.append((i,j,'E'))
            # Whatever is left
            else:
                l = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                notes.append((i,j,np.random.choice(l)))
            
    return notes            
                
                    
    
            
def specify_notes(spacing_parameter, starting_positions, detected):
    ''' This function combines all above functions and returns final output as a dictionary consisting of coordinates of each note as key and value as what the note is'''
    treble_clef, bass_clef = get_clef_positions(starting_positions)
    clef_dict = belongs_which_clef(treble_clef, bass_clef, spacing_parameter, starting_positions, detected)    
    notes = create_notes_dict(clef_dict, treble_clef, bass_clef, spacing_parameter)
    return notes
