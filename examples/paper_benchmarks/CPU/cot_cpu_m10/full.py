import quimb.tensor as qtn
import cotengra as ctg
import time
import numpy as np

from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear

import pandas as pd
import numpy as np

def read_cotengra_file(filename):
    df = pd.read_csv(filename, sep=' ', header = None)
    tensors = []
    for i in range(len(df[0])):
        # print(df[3][i])
        tens_data = df[3][i].replace("[","").replace("]","").replace("'","")
        tens_data = [complex(s) for s in tens_data.split(',')]
        tens_shape = df[2][i].replace("[","").replace("]","").replace("'","")
        tens_shape = [int(s) for s in tens_shape.split(',')]
        tens_tags = df[0][i].replace("[","").replace("]","").replace("'","")
        tens_tags = [str(s) for s in tens_tags.split(',')]
        tens_inds = df[1][i].replace("[","").replace("]","").replace("'","")
        tens_inds = [str(s) for s in tens_inds.split(',')]
        data = np.array(tens_data).reshape(tens_shape)
        inds = tens_inds
        tags = tens_tags
        tensors.append(qtn.Tensor(data, inds, tags))
    return qtn.TensorNetwork(tensors)

m10_ssa_path = ((81, 185), (127, 322), (82, 319), (99, 324), (323, 325), (304, 326), (261, 291), (9, 328), (249, 329), (206, 330), (327, 331), (69, 70), (113, 272), (333, 334), (37, 97), (298, 336), (335, 337), (314, 338), (332, 339), (230, 282), (5, 290), (341, 342), (13, 343), (235, 263), (344, 345), (169, 251), (213, 347), (346, 348), (162, 199), (207, 350), (349, 351), (170, 352), (161, 353), (340, 354), (154, 155), (355, 356), (98, 320), (120, 268), (358, 359), (357, 360), (114, 115), (361, 362), (128, 129), (363, 364), (121, 122), (365, 366), (44, 45), (310, 368), (367, 369), (76, 77), (370, 371), (218, 253), (240, 373), (16, 265), (374, 375), (10, 285), (376, 377), (247, 279), (259, 379), (200, 236), (380, 381), (378, 382), (208, 214), (383, 384), (372, 385), (163, 164), (386, 387), (123, 124), (388, 389), (95, 107), (271, 391), (63, 64), (32, 393), (392, 394), (311, 395), (390, 396), (148, 149), (397, 398), (108, 109), (399, 400), (38, 39), (305, 402), (401, 403), (71, 72), (404, 405), (86, 186), (132, 407), (54, 87), (100, 409), (408, 410), (24, 411), (406, 412), (175, 176), (413, 414), (18, 255), (267, 416), (221, 243), (417, 418), (14, 287), (419, 420), (237, 241), (215, 219), (422, 423), (421, 424), (1, 276), (223, 289), (426, 427), (226, 428), (190, 194), (231, 430), (429, 431), (6, 283), (432, 433), (425, 434), (201, 209), (435, 436), (415, 437), (171, 172), (438, 439), (156, 157), (440, 441), (116, 117), (442, 443), (50, 51), (315, 445), (444, 446), (89, 141), (187, 448), (101, 135), (57, 450), (449, 451), (179, 180), (452, 453), (447, 454), (26, 317), (455, 456), (55, 56), (457, 458), (25, 316), (459, 460), (22, 312), (461, 462), (46, 47), (463, 464), (83, 84), (465, 466), (52, 53), (467, 468), (20, 307), (469, 470), (88, 140), (177, 472), (134, 178), (133, 474), (473, 475), (471, 476), (299, 306), (477, 478), (33, 34), (479, 480), (130, 131), (174, 482), (139, 173), (85, 484), (483, 485), (481, 486), (295, 300), (487, 488), (118, 119), (321, 490), (75, 137), (160, 492), (491, 493), (96, 318), (112, 495), (68, 184), (67, 497), (496, 498), (494, 499), (42, 43), (303, 309), (501, 502), (500, 503), (73, 74), (504, 505), (258, 260), (8, 278), (507, 508), (205, 248), (509, 510), (198, 246), (511, 512), (152, 153), (513, 514), (506, 515), (110, 111), (516, 517), (106, 183), (62, 519), (31, 61), (94, 521), (520, 522), (518, 523), (21, 297), (308, 525), (524, 526), (80, 138), (125, 528), (167, 168), (126, 530), (529, 531), (48, 49), (78, 79), (533, 534), (532, 535), (527, 536), (4, 275), (281, 538), (229, 257), (539, 540), (193, 245), (541, 542), (12, 262), (250, 544), (212, 234), (545, 546), (543, 547), (197, 204), (548, 549), (537, 550), (158, 159), (551, 552), (146, 147), (553, 554), (35, 36), (302, 556), (555, 557), (65, 66), (558, 559), (23, 313), (301, 561), (560, 562), (40, 41), (19, 564), (563, 565), (58, 59), (182, 567), (28, 93), (103, 569), (568, 570), (294, 571), (566, 572), (29, 30), (296, 574), (573, 575), (136, 145), (144, 577), (60, 105), (104, 579), (578, 580), (576, 581), (27, 293), (582, 583), (92, 102), (270, 585), (584, 586), (142, 143), (587, 588), (181, 269), (90, 91), (590, 591), (292, 592), (589, 593), (189, 225), (244, 595), (256, 273), (596, 597), (2, 277), (598, 599), (228, 233), (192, 196), (601, 602), (600, 603), (7, 284), (15, 605), (239, 264), (606, 607), (217, 252), (608, 609), (604, 610), (203, 211), (611, 612), (594, 613), (165, 166), (614, 615), (150, 151), (616, 617), (489, 618), (220, 254), (242, 620), (17, 266), (621, 622), (11, 286), (623, 624), (232, 238), (210, 216), (626, 627), (625, 628), (0, 288), (222, 274), (630, 631), (188, 632), (191, 227), (224, 634), (633, 635), (3, 280), (636, 637), (629, 638), (619, 639), (195, 640), (202, 641))
m10_linear_path = ssa_to_linear(m10_ssa_path)

tn = read_cotengra_file("m10.cotengra")
tn.astype_('complex64')
#print(tn)

start = time.time()
res = tn.contract(all, optimize=m10_linear_path)
end = time.time()
print("time = " + str(end-start))
#info = tn.contract(all, optimize=m10_linear_path, get='path-info', output_inds=[])
