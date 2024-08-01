from tfpcbpggsz.PhaseSpace import *

# Define the phase space
m_mother = 1.864
m_daughter1 = 0.55
m_daughter2 = 0.139
m_daughter3 = 0.139

phsp = DalitzPhaseSpace(m_daughter1, m_daughter2, m_daughter3, m_mother)
print(phsp)