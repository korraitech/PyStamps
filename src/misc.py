#########################################################################
#   Copyright 2025 - 2025, KorrAI                                       #
#   This program is free software: you can redistribute it and/or       #
#   modify it under the terms of the GNU General Public License as      #
#   published by the Free Software Foundation, either version 3 of      #
#   the License, or (at your option) any later version.                 #
#                                                                       #
#   This program is distributed in the hope that it will be useful,     #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of      #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
#   GNU General Public License for more details.                        #
#                                                                       #
#   You should have received a copy of the GNU General Public License   #
#   along with this program. If not, see <https://www.gnu.org/licenses/>.#
#########################################################################
#                                                                       #
#   This file contains the implementation of miscellaneous utils.       #
#                                                                       #
#########################################################################


import inspect

def get_module_info()-> str:
    caller_frame = inspect.currentframe().f_back
    filename = inspect.getframeinfo(caller_frame).filename
    function_name = caller_frame.f_code.co_name
    return "{}::{}".format(filename.split('/')[-1].split('.')[0], function_name)
