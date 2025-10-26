"""
Group members: Matthew Xue, Josue Ramirez Antonio, Ruhan Xia
filename: NLPParserError.py
description: Custom exception class to deal with wrong file type extensions
"""
class NLPParserError:
    class InvalidTextFileFormat(Exception):
        pass

    @staticmethod
    def check_file_format(f):
        """ Decorator to check file type extension
            Args: function f to decorate
            Returns: wrapper fn applied to given fn f
        """
        def wrapper(*args, **kwargs):
            """ Wrapper fn to check file type extension and to raise exception if
                file with invalid extension is given
                Args: *args and *kwargs (if any) of original fn f that gets decorated
                Returns: evaluation of fn f and stored returned values/s (if any)
            """
            filepath_ext = args[1].split('.')[-1]
            if filepath_ext != 'txt' and filepath_ext != 'json':
                raise NLPParserError.InvalidTextFileFormat(
                    f"{str(args[1])}  " + "file has wrong format.\n"
                    "Please try loading a file with either '.txt' or '.json' extensions.")
            else:
                # Run the function
                val = f(*args, **kwargs)
            return val
        return wrapper
