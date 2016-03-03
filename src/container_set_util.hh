#ifndef CONTAINER_SET_UTIL_HH
#define CONTAINER_SET_UTIL_HH


namespace ContSetUt {
    template<typename T>
    T intersection(const T& cont1, const T& cont2) {
        T newCont;

        for (T::iterator it1 = cont1.begin(); it1 != cont.end(); ++it1) {
            for (T::iterator it2 = cont1.begin(); it2 != cont.end(); ++it2) {
                if (*it1 == *it2)
                    newCont.insert(*it);
            }
        }

        return T;
    }
};


#endif // CONTAINER_SET_UTIL_HH
