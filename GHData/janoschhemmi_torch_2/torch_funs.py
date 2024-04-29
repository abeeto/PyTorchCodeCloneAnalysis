##

# export
def prepare_idxs(o, shape=None):
    if o is None:
        return slice(None)
    elif is_slice(o) or isinstance(o, Integral):
        return o
    else:
        if shape is not None:
            return np.array(o).reshape(shape)
        else:
            return np.array(o)




def prepare_sel_vars_and_steps(sel_vars=None, sel_steps=None, idxs=False):
    sel_vars = prepare_idxs(sel_vars)
    sel_steps = prepare_idxs(sel_steps)
    if not is_slice(sel_vars) and not isinstance(sel_vars, Integral):
        if is_slice(sel_steps) or isinstance(sel_steps, Integral):
            _sel_vars = [sel_vars, sel_vars.reshape(1, -1)]
        else:
            _sel_vars = [sel_vars.reshape(-1, 1), sel_vars.reshape(1, -1, 1)]
    else:
        _sel_vars = [sel_vars] * 2
    if not is_slice(sel_steps) and not isinstance(sel_steps, Integral):
        if is_slice(sel_vars) or isinstance(sel_vars, Integral):
            _sel_steps = [sel_steps, sel_steps.reshape(1, -1)]
        else:
            _sel_steps = [sel_steps.reshape(1, -1), sel_steps.reshape(1, 1, -1)]
    else:
        _sel_steps = [sel_steps] * 2
    if idxs:
        n_dim = np.sum([isinstance(o, np.ndarray) for o in [sel_vars, sel_steps]])
        idx_shape = (-1,) + (1,) * n_dim
        return _sel_vars, _sel_steps, idx_shape
    else:
        return _sel_vars[0], _sel_steps[0]


def apply_sliding_window(
        data,  # and array-like object with the input data
        window_len: int | list,  # sliding window length. When using a list, use negative numbers and 0.
        horizon: int | list = 0,  # horizon
        x_vars: int | list | None = None,  # indices of the independent variables
        y_vars: int | list | None = None,
        # indices of the dependent variables (target). [] means no y will be created. None means all variables.
):
    "Applies a sliding window on an array-like input to generate a 3d X (and optionally y)"

    ## test
    data = subset
    window_len = 21
    x_vars = ["1","2","3","4","5"]
    horizon = 0

    if isinstance(data, pd.DataFrame): data = data.to_numpy()
    if isinstance(window_len, list):
        assert np.max(window_len) == 0
        x_steps = abs(np.min(window_len)) + np.array(window_len)
        window_len = abs(np.min(window_len)) + 1
    else:
        x_steps = None

    # 5 layer of ts shape = 5 / n /21
    X_data_windowed = np.lib.stride_tricks.sliding_window_view(data, window_len, axis=0)
    X_data_windowed = np.lib.stride_tricks.sliding_window_view(data, (len(x_vars),window_len ))

    # X
    sel_x_vars, sel_x_steps = prepare_sel_vars_and_steps(x_vars, x_steps)
    if horizon == 0:
        X = X_data_windowed[:, sel_x_vars, sel_x_steps]
    else:
        X = X_data_windowed[:-np.max(horizon):, sel_x_vars, sel_x_steps]
    if x_vars is not None and isinstance(x_vars, Integral):
        X = X[:, None]  # keep 3 dim

    # y
    if y_vars == []:
        y = None
    else:
        if isinstance(horizon, Integral) and horizon == 0:
            y = data[-len(X):, y_vars]
        else:
            y_data_windowed = np.lib.stride_tricks.sliding_window_view(data, np.max(horizon) + 1, axis=0)[-len(X):]
            y_vars, y_steps = prepare_sel_vars_and_steps(y_vars, horizon)
            y = np.squeeze(y_data_windowed[:, y_vars, y_steps])
    return X, y



def get_splits(o, n_splits:int=1, valid_size:float=0.2, test_size:float=0., train_only:bool=False, train_size:Union[None, float, int]=None, balance:bool=False,
               shuffle:bool=True, stratify:bool=True, check_splits:bool=True, random_state:Union[None, int]=None, show_plot:bool=True, verbose:bool=False):
    '''Arguments:
        o            : object to which splits will be applied, usually target.
        n_splits     : number of folds. Must be an int >= 1.
        valid_size   : size of validation set. Only used if n_splits = 1. If n_splits > 1 valid_size = (1. - test_size) / n_splits.
        test_size    : size of test set. Default = 0.
        train_only   : if True valid set == train set. This may be useful for debugging purposes.
        train_size   : size of the train set used. Default = None (the remainder after assigning both valid and test).
                        Useful for to get learning curves with different train sizes or get a small batch to debug a neural net.
        balance      : whether to balance data so that train always contain the same number of items per class.
        shuffle      : whether to shuffle data before splitting into batches. Note that the samples within each split will be shuffle.
        stratify     : whether to create folds preserving the percentage of samples for each class.
        check_splits : whether to perform leakage and completion checks.
        random_state : when shuffle is True, random_state affects the ordering of the indices. Pass an int for reproducible output.
        show_plot    : plot the split distribution
    '''
    if n_splits == 1 and valid_size == 0. and  test_size == 0.: train_only = True
    if balance: stratify = True
    splits = TrainValidTestSplitter(n_splits, valid_size=valid_size, test_size=test_size, train_only=train_only, stratify=stratify,
                                      balance=balance, shuffle=shuffle, random_state=random_state, verbose=verbose)(o)
    if check_splits:
        if train_only or (n_splits == 1 and valid_size == 0): print('valid == train')
        elif n_splits > 1:
            for i in range(n_splits):
                leakage_finder([*splits[i]], verbose=True)
                cum_len = 0
                for split in splits[i]: cum_len += len(split)
                if not balance: assert len(o) == cum_len, f'len(o)={len(o)} while cum_len={cum_len}'
        else:
            leakage_finder([splits], verbose=True)
            cum_len = 0
            if not isinstance(splits[0], Integral):
                for split in splits: cum_len += len(split)
            else: cum_len += len(splits)
            if not balance: assert len(o) == cum_len, f'len(o)={len(o)} while cum_len={cum_len}'
    if train_size is not None and train_size != 1: # train_size=1 legacy
        if n_splits > 1:
            splits = list(splits)
            for i in range(n_splits):
                splits[i] = list(splits[i])
                if isinstance(train_size, Integral):
                    n_train_samples = train_size
                elif train_size > 0 and train_size < 1:
                    n_train_samples = int(len(splits[i][0]) * train_size)
                splits[i][0] = L(np.random.choice(splits[i][0], n_train_samples, False).tolist())
                if train_only:
                    if valid_size != 0: splits[i][1] = splits[i][0]
                    if test_size != 0: splits[i][2] = splits[i][0]
                splits[i] = tuple(splits[i])
            splits = tuple(splits)
        else:
            splits = list(splits)
            if isinstance(train_size, Integral):
                n_train_samples = train_size
            elif train_size > 0 and train_size < 1:
                n_train_samples = int(len(splits[0]) * train_size)
            splits[0] = L(np.random.choice(splits[0], n_train_samples, False).tolist())
            if train_only:
                if valid_size != 0: splits[1] = splits[0]
                if test_size != 0: splits[2] = splits[0]
            splits = tuple(splits)
    if show_plot: plot_splits(splits)
    return splits


# Cell
def TrainValidTestSplitter(n_splits:int=1, valid_size:Union[float, int]=0.2, test_size:Union[float, int]=0., train_only:bool=False,
                           stratify:bool=True, balance:bool=False, shuffle:bool=True, random_state:Union[None, int]=None, verbose:bool=False, **kwargs):
    "Split `items` into random train, valid (and test optional) subsets."

    if not shuffle and stratify and not train_only:
        pv('stratify set to False because shuffle=False. If you want to stratify set shuffle=True', verbose)
        stratify = False

    def _inner(o, **kwargs):
        if stratify:
            _, unique_counts = np.unique(o, return_counts=True)
            if np.min(unique_counts) >= 2 and np.min(unique_counts) >= n_splits: stratify_ = stratify
            elif np.min(unique_counts) < n_splits:
                stratify_ = False
                pv(f'stratify set to False as n_splits={n_splits} cannot be greater than the min number of members in each class ({np.min(unique_counts)}).',
                   verbose)
            else:
                stratify_ = False
                pv('stratify set to False as the least populated class in o has only 1 member, which is too few.', verbose)
        else: stratify_ = False
        vs = 0 if train_only else 1. / n_splits if n_splits > 1 else int(valid_size * len(o)) if isinstance(valid_size, float) else valid_size
        if test_size:
            ts = int(test_size * len(o)) if isinstance(test_size, float) else test_size
            train_valid, test = train_test_split(range(len(o)), test_size=ts, stratify=o if stratify_ else None, shuffle=shuffle,
                                                 random_state=random_state, **kwargs)
            test = toL(test)
            if shuffle: test = random_shuffle(test, random_state)
            if vs == 0:
                train, _ = RandomSplitter(0, seed=random_state)(o[train_valid])
                train = toL(train)
                if balance: train = train[balance_idx(o[train], random_state=random_state)]
                if shuffle: train = random_shuffle(train, random_state)
                train_ = L(L([train]) * n_splits) if n_splits > 1 else train
                valid_ = L(L([train]) * n_splits) if n_splits > 1 else train
                test_ = L(L([test]) * n_splits) if n_splits > 1 else test
                if n_splits > 1:
                    return [split for split in itemify(train_, valid_, test_)]
                else:
                    return train_, valid_, test_
            elif n_splits > 1:
                if stratify_:
                    splits = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(np.arange(len(train_valid)), o[train_valid])
                else:
                    splits = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(np.arange(len(train_valid)))
                train_, valid_ = L([]), L([])
                for train, valid in splits:
                    train, valid = toL(train), toL(valid)
                    if balance: train = train[balance_idx(o[train], random_state=random_state)]
                    if shuffle:
                        train = random_shuffle(train, random_state)
                        valid = random_shuffle(valid, random_state)
                    train_.append(L(L(train_valid)[train]))
                    valid_.append(L(L(train_valid)[valid]))
                test_ = L(L([test]) * n_splits)
                return [split for split in itemify(train_, valid_, test_)]
            else:
                train, valid = train_test_split(range(len(train_valid)), test_size=vs, random_state=random_state,
                                                stratify=o[train_valid] if stratify_ else None, shuffle=shuffle, **kwargs)
                train, valid = toL(train), toL(valid)
                if balance: train = train[balance_idx(o[train], random_state=random_state)]
                if shuffle:
                    train = random_shuffle(train, random_state)
                    valid = random_shuffle(valid, random_state)
                return (L(L(train_valid)[train]), L(L(train_valid)[valid]),  test)
        else:
            if vs == 0:
                train, _ = RandomSplitter(0, seed=random_state)(o)
                train = toL(train)
                if balance: train = train[balance_idx(o[train], random_state=random_state)]
                if shuffle: train = random_shuffle(train, random_state)
                train_ = L(L([train]) * n_splits) if n_splits > 1 else train
                valid_ = L(L([train]) * n_splits) if n_splits > 1 else train
                if n_splits > 1:
                    return [split for split in itemify(train_, valid_)]
                else:
                    return (train_, valid_)
            elif n_splits > 1:
                if stratify_: splits = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(np.arange(len(o)), o)
                else: splits = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state).split(np.arange(len(o)))
                train_, valid_ = L([]), L([])
                for train, valid in splits:
                    train, valid = toL(train), toL(valid)
                    if balance: train = train[balance_idx(o[train], random_state=random_state)]
                    if shuffle:
                        train = random_shuffle(train, random_state)
                        valid = random_shuffle(valid, random_state)
                    if not isinstance(train, (list, L)):  train = train.tolist()
                    if not isinstance(valid, (list, L)):  valid = valid.tolist()
                    train_.append(L(train))
                    valid_.append(L(L(valid)))
                return [split for split in itemify(train_, valid_)]
            else:
                train, valid = train_test_split(range(len(o)), test_size=vs, random_state=random_state, stratify=o if stratify_ else None,
                                                shuffle=shuffle, **kwargs)
                train, valid = toL(train), toL(valid)
                if balance: train = train[balance_idx(o[train], random_state=random_state)]
                return train, valid
    return _inner

# Cell
def plot_splits(splits):
    _max = 0
    _splits = 0
    for i, split in enumerate(splits):
        if is_listy(split[0]):
            for j, s in enumerate(split):
                _max = max(_max, array(s).max())
                _splits += 1
        else:
            _max = max(_max, array(split).max())
            _splits += 1
    _splits = [splits] if not is_listy(split[0]) else splits
    v = np.zeros((len(_splits), _max + 1))
    for i, split in enumerate(_splits):
        if is_listy(split[0]):
            for j, s in enumerate(split):
                v[i, s] = 1 + j
        else: v[i, split] = 1 + i
    vals = np.unique(v)
    plt.figure(figsize=(16, len(_splits)/2))
    if len(vals) == 1:
        v = np.ones((len(_splits), _max + 1))
        plt.pcolormesh(v, color='blue')
        legend_elements = [Patch(facecolor='blue', label='Train')]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        colors = L(['gainsboro', 'blue', 'limegreen', 'red'])[vals]
        cmap = LinearSegmentedColormap.from_list('', colors)
        plt.pcolormesh(v, cmap=cmap)
        legend_elements = L([
            Patch(facecolor='gainsboro', label='None'),
            Patch(facecolor='blue', label='Train'),
            Patch(facecolor='limegreen', label='Valid'),
            Patch(facecolor='red', label='Test')])[vals]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Split distribution')
    plt.yticks(ticks=np.arange(.5, len(_splits)+.5, 1.0), labels=np.arange(1, len(_splits)+1, 1.0).astype(int))
    plt.gca().invert_yaxis()
    plt.show()

# Cell
def get_splits(o, n_splits:int=1, valid_size:float=0.2, test_size:float=0., train_only:bool=False, train_size:Union[None, float, int]=None, balance:bool=False,
               shuffle:bool=True, stratify:bool=True, check_splits:bool=True, random_state:Union[None, int]=None, show_plot:bool=True, verbose:bool=False):
    '''Arguments:
        o            : object to which splits will be applied, usually target.
        n_splits     : number of folds. Must be an int >= 1.
        valid_size   : size of validation set. Only used if n_splits = 1. If n_splits > 1 valid_size = (1. - test_size) / n_splits.
        test_size    : size of test set. Default = 0.
        train_only   : if True valid set == train set. This may be useful for debugging purposes.
        train_size   : size of the train set used. Default = None (the remainder after assigning both valid and test).
                        Useful for to get learning curves with different train sizes or get a small batch to debug a neural net.
        balance      : whether to balance data so that train always contain the same number of items per class.
        shuffle      : whether to shuffle data before splitting into batches. Note that the samples within each split will be shuffle.
        stratify     : whether to create folds preserving the percentage of samples for each class.
        check_splits : whether to perform leakage and completion checks.
        random_state : when shuffle is True, random_state affects the ordering of the indices. Pass an int for reproducible output.
        show_plot    : plot the split distribution
    '''
    if n_splits == 1 and valid_size == 0. and  test_size == 0.: train_only = True
    if balance: stratify = True
    splits = TrainValidTestSplitter(n_splits, valid_size=valid_size, test_size=test_size, train_only=train_only, stratify=stratify,
                                      balance=balance, shuffle=shuffle, random_state=random_state, verbose=verbose)(o)
    if check_splits:
        if train_only or (n_splits == 1 and valid_size == 0): print('valid == train')
        elif n_splits > 1:
            for i in range(n_splits):
                leakage_finder([*splits[i]], verbose=True)
                cum_len = 0
                for split in splits[i]: cum_len += len(split)
                if not balance: assert len(o) == cum_len, f'len(o)={len(o)} while cum_len={cum_len}'
        else:
            leakage_finder([splits], verbose=True)
            cum_len = 0
            if not isinstance(splits[0], Integral):
                for split in splits: cum_len += len(split)
            else: cum_len += len(splits)
            if not balance: assert len(o) == cum_len, f'len(o)={len(o)} while cum_len={cum_len}'
    if train_size is not None and train_size != 1: # train_size=1 legacy
        if n_splits > 1:
            splits = list(splits)
            for i in range(n_splits):
                splits[i] = list(splits[i])
                if isinstance(train_size, Integral):
                    n_train_samples = train_size
                elif train_size > 0 and train_size < 1:
                    n_train_samples = int(len(splits[i][0]) * train_size)
                splits[i][0] = L(np.random.choice(splits[i][0], n_train_samples, False).tolist())
                if train_only:
                    if valid_size != 0: splits[i][1] = splits[i][0]
                    if test_size != 0: splits[i][2] = splits[i][0]
                splits[i] = tuple(splits[i])
            splits = tuple(splits)
        else:
            splits = list(splits)
            if isinstance(train_size, Integral):
                n_train_samples = train_size
            elif train_size > 0 and train_size < 1:
                n_train_samples = int(len(splits[0]) * train_size)
            splits[0] = L(np.random.choice(splits[0], n_train_samples, False).tolist())
            if train_only:
                if valid_size != 0: splits[1] = splits[0]
                if test_size != 0: splits[2] = splits[0]
            splits = tuple(splits)
    if show_plot: plot_splits(splits)
    return splits

# Cell
def TSSplitter(valid_size:Union[int, float]=0.2, test_size:Union[int, float]=0., show_plot:bool=True):
    "Create function that splits `items` between train/val with `valid_size` without shuffling data."
    def _inner(o):
        valid_cut = valid_size if isinstance(valid_size, Integral) else int(round(valid_size * len(o)))
        if test_size:
            test_cut = test_size if isinstance(test_size, Integral) else int(round(test_size * len(o)))
        idx = np.arange(len(o))
        if test_size:
            splits = L(idx[:-valid_cut - test_cut].tolist()), L(idx[-valid_cut - test_cut: - test_cut].tolist()), L(idx[-test_cut:].tolist())
        else:
            splits = L(idx[:-valid_cut].tolist()), L(idx[-valid_cut:].tolist())
        if show_plot:
            if len(o) > 1_000_000:
                warnings.warn('the splits are too large to be plotted')
            else:
                plot_splits(splits)
        return splits
    return _inner

TimeSplitter = TSSplitter

# Cell
def get_predefined_splits(*xs):
    '''xs is a list with X_train, X_valid, ...'''
    splits_ = []
    start = 0
    for x in xs:
        splits_.append(L(list(np.arange(start, start + len(x)))))
        start += len(x)
    return tuple(splits_)

def combine_split_data(xs, ys=None):
    '''xs is a list with X_train, X_valid, .... ys is None or a list with y_train, y_valid, .... '''
    xs = [to3d(x) for x in xs]
    splits = get_predefined_splits(*xs)
    if ys is None: return concat(*xs), None, splits
    else: return concat(*xs), concat(*ys), splits

# Cell
def get_splits_len(splits):
    _len = []
    for split in splits:
        if isinstance(split[0], (list, L, tuple)):  _len.append([len(s) for s in split])
        else: _len.append(len(split))
    return _len