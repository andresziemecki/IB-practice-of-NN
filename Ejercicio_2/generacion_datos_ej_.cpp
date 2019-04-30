#include <iostream>
#include <fstream> 

class V_t{
    int *v;
    int size;
    public:
        V_t(int i): size(i){v= new int[i];
                            int j;
                            for (j=0; j < i; j++){
                                v[j] = 1;
                            }
                        }
        ~V_t(){delete v;}
        V_t(const V_t &H);
        void print_V(void) const;
        friend void recursion(V_t vect, int position);
};
/* Separado por espacios
void V_t::print_V(void) const{
    int i;
    for (i=0; i< size; i++)
    {
        if (v[i] == 1)
        std::cout << ' ' << v[i] << "  ";
        else
        std::cout << v[i] << "  ";
    }
    std::cout << std::endl;
    return; 
}
*/

// Separados por coma
void V_t::print_V(void) const{
    int i;
    for (i=0; i< size; i++)
    {
        if (i==(size-1))
            std::cout << v[i];
        else
            std::cout << v[i] << ',';
    }
    std::cout << std::endl;
    return; 
}
V_t::V_t(const V_t &H){
    this->v = new int[H.size];
    if (!v)
        std::cout << "Memory allocation failed\n";
    int i;
    size = H.size;
    for (i=0; i< size; i++){
        this->v[i] = H.v[i];
    }
}

void recursion(V_t vect, int position){
    if (position < vect.size-1)
        recursion(vect, position + 1);
    else{
        vect.print_V();
        vect.v[position] = -1;
        vect.print_V();
        return;
    }
    vect.v[position] = -1;
    recursion(vect, position +1);
}

//*******************PROGRAMA**********************//
int main(void){
    V_t X(8); // numero de columnas
    recursion(X,0);
    return 0;
}



 







