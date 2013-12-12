def progress(counter=True, **kwargs):
    import progressbar as pb
    try:
        widgets = kwargs.pop('widgets')
    except KeyError:
        # TODO: make work when maxval is unknown
        if counter:
            class CommaProgress(pb.Widget):
                def update(self, pbar):
                    return '{:,} of {:,}'.format(pbar.currval, pbar.maxval)
            widgets = [' ', CommaProgress(), ' (', pb.Percentage(), ') ']
        else:
            widgets = [' ', pb.Percentage(), ' ']
        widgets.extend([pb.Bar(), ' ', pb.ETA()])
    return pb.ProgressBar(widgets=widgets, **kwargs)

