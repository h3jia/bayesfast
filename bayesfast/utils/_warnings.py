import warnings
from warnings import WarningMessage, _showwarnmsg_impl


class SamplingProgess(UserWarning):
    pass


__all__ = ['SamplingProgess', 'formatwarning_chain', 'showwarning_chain']


def _showwarning_chain(message, category, filename, lineno, file=None, 
                       line=None, chain=None):
    """Hook to write a warning to a file; replace if you like."""
    if category is SamplingProgess and chain is not None:
        print("CHAIN #{}: ".format(chain) + str(message))
    else:
        msg = WarningMessage(message, category, filename, lineno, file, line)
        _showwarnmsg_impl(msg)


def _formatwarning_chain(message, category, filename, lineno, line=None, 
                         chain=None):
    """Function to format a warning the standard way."""
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_implc(msg, chain)


def _formatwarnmsg_implc(msg, chain):
    if chain is None:
        s =  ("%s:%s: %s: %s\n"
          % (msg.filename, msg.lineno, msg.category.__name__,
             msg.message))
    else:
        s =  ("%s:%s: %s from CHAIN #%s: %s\n"
              % (msg.filename, msg.lineno, msg.category.__name__, chain, 
                 msg.message))

    if msg.line is None:
        try:
            import linecache
            line = linecache.getline(msg.filename, msg.lineno)
        except Exception:
            # When a warning is logged during Python shutdown, linecache
            # and the import machinery don't work anymore
            line = None
            linecache = None
    else:
        line = msg.line
    if line:
        line = line.strip()
        s += "  %s\n" % line

    if msg.source is not None:
        try:
            import tracemalloc
            tb = tracemalloc.get_object_traceback(msg.source)
        except Exception:
            # When a warning is logged during Python shutdown, tracemalloc
            # and the import machinery don't work anymore
            tb = None

        if tb is not None:
            s += 'Object allocated at (most recent call first):\n'
            for frame in tb:
                s += ('  File "%s", lineno %s\n'
                      % (frame.filename, frame.lineno))

                try:
                    if linecache is not None:
                        line = linecache.getline(frame.filename, frame.lineno)
                    else:
                        line = None
                except Exception:
                    line = None
                if line:
                    line = line.strip()
                    s += '    %s\n' % line
    return s


def formatwarning_chain(chain=0):
    return lambda *args, **kwargs: _formatwarning_chain(*args, **kwargs, 
                                                        chain=chain)


def showwarning_chain(chain=0):
    return lambda *args, **kwargs: _showwarning_chain(*args, **kwargs, 
                                                      chain=chain)
