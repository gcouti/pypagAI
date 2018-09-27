import io
import os
import pickle
import zipfile


class ModelDumper(object):
    """Helper to persist a model using pickle.

    Parameters
    ----------
    estimator : estimator object
                An estimator object implementing `fit`
    """

    def __init__(self, estimator):
        super(ModelDumper, self).__init__()
        self.estimator_ = estimator

    def dump(self, file):
        """Persists model.

        Parameters
        ----------
        file : str or IOBase
               when str is given, file path is expected
        """
        if isinstance(file, str):
            f = open(file, 'wb')
        elif isinstance(file, io.IOBase):
            f = file
        else:
            raise Exception("Unexpected type.")

        s = pickle.dumps(self.estimator_)

        f.write(s)
        f.close()


class ModelLoader(object):
    """
    Helper to load a persisted a model using pickle.
    """

    def __init__(self):
        super(ModelLoader, self).__init__()

    def load(self, file):
        """Loads model.

        Parameters
        ----------
        file : str or IOBase
               when str is given, file path is expected
        """
        if isinstance(file, str):
            base_file, ext = os.path.splitext(file)

            if ext == '.zip':
                zf = zipfile.ZipFile(file, 'r')
                f = zf.open(zf.namelist()[0])
            else:
                f = open(file, 'rb')
        elif isinstance(file, io.IOBase):
            f = file
        else:
            raise Exception("Unexpected type.")

        estimator_ = pickle.loads(f.read())

        f.close()

        return estimator_
