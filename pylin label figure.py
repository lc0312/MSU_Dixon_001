# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:28:19 2024

@author: ozturklab
"""

from matplotlib import image 
from matplotlib import pyplot as plt 
  

data = image.imread(r"C:\Users\ozturklab\Desktop\Chao_Liu_Nanowire\Articles\2\Figures\20241115_132538 32.jpg") 
  
plt.axis('off')
plt.plot([], [], alpha=0, label=' d)')
#plt.legend(frameon=False, loc=1, fontsize=14,labelcolor='white')

plt.imshow(data)
#plt.show ()
 

plt.savefig("test.png",dpi=600, bbox_inches='tight',pad_inches = 0)


