// © Copyright 2008 Vladimir Frolov, Moscow State University Graphics & Media Lab
#pragma once

#include <vector>
#include <assert.h>

namespace MGML_MEMORY
{

//
//
template<class T, int threadId = 0>
class FastList
{
public:

  class Elem
  {
  public:

    friend class FastList;
  protected:
    T m_data;
    Elem* m_next;
  };

  class iterator
  {
  public:

    iterator() {m_elem = 0;}
    iterator(Elem* a_elem) { m_elem = a_elem; }

    bool operator==(const iterator& a_it) const { return (a_it.m_elem == m_elem); }
    bool operator!=(const iterator& a_it) const { return !(*this == a_it); }

    iterator operator++()
    {
      assert(m_elem!=0);
      m_elem = m_elem->m_next;
      return iterator(m_elem);
    }

    iterator operator++(int)
    {
      iterator prev = *this;
      ++(*this);
      return prev;
    }

    T* operator->() const
    {
      assert(m_elem!=0);
      return &(m_elem->m_data);
    }

    T& operator*()
    {
      assert(m_elem!=0);
      return m_elem->m_data;
    }

    friend class const_iterator;
    friend class FastList;

  protected:
    Elem* m_elem;
  };


  class const_iterator
  {
  public:

    const_iterator() {m_elem = 0;}
    const_iterator(const Elem* a_elem) { m_elem = a_elem; }
    const_iterator(const iterator& p) { m_elem = p.m_elem; }

    bool operator==(const_iterator a_it) const { return (a_it.m_elem == m_elem); }
    bool operator!=(const_iterator a_it) const { return !(*this == a_it); }

    const_iterator operator++()
    {
      assert(m_elem!=0);
      m_elem = m_elem->m_next;
      return const_iterator(m_elem);
    }

    const_iterator operator++(int)
    {
      iterator prev = *this;
      ++(*this);
      return prev;
    }

    const T* operator->() const
    {
      assert(m_elem!=0);
      return &(m_elem->m_data);
    }

    const T& operator*() const
    {
      assert(m_elem!=0);
      return m_elem->m_data;
    }


  protected:
    const Elem* m_elem;
  };


  FastList() { m_head = 0; m_size=0; }

  inline bool empty() const {return (m_head == 0);}
  inline void Reset() { m_head = 0;m_size = 0;} // you must Reset all your lists after Free call, or delete them

  inline const_iterator begin() const { return const_iterator(m_head);}
  inline const_iterator end() const { return const_iterator(0);}

  inline iterator begin() { return iterator(m_head);}
  inline iterator end() { return iterator(0);}

  inline void push_front(const T& a_data)
  {
    Elem *newElem = m_freeList;

#ifndef NDEBUG
    if(newElem==NULL)
      std::cerr << "out of memory for " << typeid(this).name() << std::endl;
#endif
    assert(newElem!=0);

    m_freeList = newElem->m_next;
    newElem->m_next = m_head;
    newElem->m_data = a_data;
    m_head = newElem;
    m_size++;
  }

  inline void pop_front()
  {
    assert(m_head != 0);

    Elem* next = m_head->m_next;
    m_head->m_next = m_freeList;
    m_freeList = m_head;
    m_head = next;
    m_size--;
  }

  void clear()
  {
    while(m_head != 0)
      pop_front();

    m_size = 0;
  }

  size_t size() const
  {
    return m_size;
  }

  inline void erase_next(iterator p)
  {
     Elem* next = p.m_elem->m_next;
     assert(next!=NULL);

     // (p -> next -> next-next) =>  (p -> next-next)
     //
     p.m_elem->m_next = next->m_next;

     // add next to free memory list
     //
     next->m_next = m_freeList;
     m_freeList = next;
     m_size--;
  }

  static void Allocate(int a_size)
  {
    m_memory = (Elem*)malloc(a_size*sizeof(Elem)); //_aligned_malloc(a_size*sizeof(Elem), 16); //new Elem[a_size];

    if (m_memory == nullptr)
      RUN_TIME_ERROR("FastList::Allocate have failed!");

    m_totalMemory = a_size*sizeof(Elem);
    T defaultData;

    //if(TypesEqual<T,int>::RET)
      //defaultData = 0;

    for(int i=0;i<a_size-1;i++)
    {
      //if(!TypesEqual<T,int>::RET)
        //m_memory[i].m_data = defaultData;
      m_memory[i].m_next = m_memory + i + 1;
    }
    m_memory[a_size-1].m_next = 0;

    m_freeList = m_memory;

  }

  static void Free()
  {
    //memset(m_memory,0xBB,m_totalMemory);
    //delete [] m_memory;
    free(m_memory);
    m_memory = 0;
  }

  static int GetFreeMemoryInElements()
  {
    int counter = 0;
    Elem* elem = m_freeList;

    while(elem->m_next!=NULL)
    {
      counter++;
      elem = elem->m_next;
    }

    return counter;
  }

  static float GetPercentOfFreeMemory()
  {
    float free_mem = (float)GetFreeMemoryInElements()*sizeof(Elem);
    float total_mem = (float)m_totalMemory;
    return free_mem/total_mem;
  }

protected:
  static Elem* m_memory;
  static Elem* m_freeList;
  static int m_totalMemory;

  Elem* m_head;
  int m_size;
};

template<class T, int threadId> typename FastList<T,threadId>::Elem* FastList<T,threadId>::m_memory = NULL;
template<class T, int threadId> typename FastList<T,threadId>::Elem* FastList<T,threadId>::m_freeList = NULL;
template<class T, int threadId> int FastList<T,threadId>::m_totalMemory = 0;

};

